"""LightGBM-like training job routines."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from importlib import import_module


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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

    rows = conn.execute(
        """
        SELECT label_id, feature_name, feature_value, target, local_date
        FROM vw_lightgbm_training_dataset
        """
    ).fetchall()
    row_count = len(rows)
    day_count = len({str(row["local_date"]) for row in rows})

    if row_count < min_labeled_rows or day_count < min_labeled_days:
        conn.execute(
            """
            UPDATE lightgbm_training_runs
            SET finished_at_utc = ?, status = ?, row_count = ?, day_count = ?, notes = ?
            WHERE id = ?
            """,
            (_utc_now(), "skipped", row_count, day_count, "training gate not met", run_id),
        )
        conn.commit()
        return None

    positive = defaultdict(list)
    negative = defaultdict(list)
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

    feature_names = sorted(weights_by_name.keys())
    model_payload: dict[str, object] = {
        "intercept": 0.0,
        "weights": [weights_by_name[name] for name in feature_names],
    }
    model_type = "lightgbm_like"
    notes = "ok"

    # Attempt real LightGBM artifact generation when the dependency is available.
    sample_targets = list(label_targets.values())
    if feature_names and sample_targets and len(set(sample_targets)) > 1:
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
