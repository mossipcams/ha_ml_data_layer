"""LightGBM-like training job routines."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from importlib import import_module
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_training_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT label_id, feature_name, feature_value, target, local_date
        FROM vw_lightgbm_training_dataset
        """
    ).fetchall()


def _row_and_day_counts(rows: list[sqlite3.Row]) -> tuple[int, int]:
    row_count = len(rows)
    day_count = len({str(row["local_date"]) for row in rows})
    return row_count, day_count


def _skip_run(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    row_count: int,
    day_count: int,
    notes: str,
) -> None:
    conn.execute(
        """
        UPDATE lightgbm_training_runs
        SET finished_at_utc = ?, status = ?, row_count = ?, day_count = ?, notes = ?
        WHERE id = ?
        """,
        (_utc_now(), "skipped", row_count, day_count, notes, run_id),
    )
    conn.commit()


def _build_training_aggregates(
    rows: list[sqlite3.Row],
) -> tuple[dict[str, float], dict[int, dict[str, float]], dict[int, int]]:
    positive: dict[str, list[float]] = defaultdict(list)
    negative: dict[str, list[float]] = defaultdict(list)
    label_feature_values: dict[int, dict[str, float]] = {}
    label_targets: dict[int, int] = {}
    for row in rows:
        label_id = int(row["label_id"])
        name = str(row["feature_name"])
        value = float(row["feature_value"])
        target = int(row["target"])
        label_feature_values.setdefault(label_id, {})[name] = value
        label_targets[label_id] = target
        if target == 1:
            positive[name].append(value)
        else:
            negative[name].append(value)

    weights_by_name: dict[str, float] = {}
    for name in sorted(set(positive) | set(negative)):
        pos_mean = sum(positive[name]) / len(positive[name]) if positive[name] else 0.0
        neg_mean = sum(negative[name]) / len(negative[name]) if negative[name] else 0.0
        weights_by_name[name] = pos_mean - neg_mean
    return weights_by_name, label_feature_values, label_targets


def _try_attach_real_lightgbm_payload(
    *,
    feature_names: list[str],
    model_payload: dict[str, Any],
    label_feature_values: dict[int, dict[str, float]],
    label_targets: dict[int, int],
) -> tuple[str, str]:
    model_type = "lightgbm_like"
    notes = "ok"
    sample_targets = list(label_targets.values())
    if not (feature_names and sample_targets and len(set(sample_targets)) > 1):
        return model_type, notes

    try:
        lightgbm = import_module("lightgbm")
        classifier = lightgbm.LGBMClassifier(
            objective="binary",
            n_estimators=16,
            learning_rate=0.1,
            random_state=42,
        )
        sample_ids = sorted(label_feature_values.keys())
        samples = [label_feature_values[label_id] for label_id in sample_ids]
        x_rows = [
            [float(sample.get(feature_name, 0.0)) for feature_name in feature_names]
            for sample in samples
        ]
        y_rows = [int(label_targets[label_id]) for label_id in sample_ids]
        classifier.fit(x_rows, y_rows)
        booster_model_str = str(classifier.booster_.model_to_string())
        if booster_model_str.strip():
            model_payload["booster_model_str"] = booster_model_str
            model_type = "lightgbm_binary_classifier"
            notes = "ok_lightgbm"
    except Exception:
        # Keep legacy payload for robust operation when dependency/runtime data is insufficient.
        pass
    return model_type, notes


def run_lightgbm_training_job(
    conn: sqlite3.Connection,
    *,
    min_labeled_rows: int = 20,
    min_labeled_days: int = 5,
) -> int | None:
    started_at = _utc_now()
    run_cur = conn.execute(
        """
        INSERT INTO lightgbm_training_runs(started_at_utc, status, row_count, day_count, notes)
        VALUES (?, ?, 0, 0, ?)
        """,
        (started_at, "started", ""),
    )
    run_id = int(run_cur.lastrowid)

    rows = _load_training_rows(conn)
    row_count, day_count = _row_and_day_counts(rows)

    if row_count < min_labeled_rows or day_count < min_labeled_days:
        _skip_run(
            conn,
            run_id=run_id,
            row_count=row_count,
            day_count=day_count,
            notes="training gate not met",
        )
        return None

    target_values = {int(row["target"]) for row in rows}
    if len(target_values) < 2:
        _skip_run(
            conn,
            run_id=run_id,
            row_count=row_count,
            day_count=day_count,
            notes="training gate not met: class balance",
        )
        return None

    weights_by_name, label_feature_values, label_targets = _build_training_aggregates(rows)
    feature_names = sorted(weights_by_name.keys())
    model_payload: dict[str, object] = {
        "intercept": 0.0,
        "weights": [weights_by_name[name] for name in feature_names],
    }
    model_type, notes = _try_attach_real_lightgbm_payload(
        feature_names=feature_names,
        model_payload=model_payload,
        label_feature_values=label_feature_values,
        label_targets=label_targets,
    )

    artifact_json = json.dumps(
        {
            "model": model_payload,
            "feature_names": feature_names,
        },
        sort_keys=True,
    )
    conn.execute(
        """
        INSERT INTO lightgbm_model_artifacts(
            run_id, created_at_utc, model_type, feature_set_version, artifact_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, _utc_now(), model_type, "v1", artifact_json),
    )
    conn.execute(
        """
        UPDATE lightgbm_training_runs
        SET finished_at_utc = ?, status = ?, row_count = ?, day_count = ?, notes = ?
        WHERE id = ?
        """,
        (_utc_now(), "completed", row_count, day_count, notes, run_id),
    )
    conn.commit()
    return run_id
