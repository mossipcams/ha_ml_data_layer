"""CLR training job routines."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _get_logistic_regression_class():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression


def _build_training_matrix(rows: list[sqlite3.Row]) -> tuple[list[str], list[list[float]], list[int], set[str]]:
    by_label: dict[int, dict] = {}
    feature_names: set[str] = set()
    day_values: set[str] = set()

    for row in rows:
        label_id = int(row["label_id"])
        feature_name = str(row["feature_name"])
        feature_value = float(row["feature_value"])
        target = int(row["target"])
        local_date = str(row["local_date"])
        feature_names.add(feature_name)
        day_values.add(local_date)
        if label_id not in by_label:
            by_label[label_id] = {
                "target": target,
                "local_date": local_date,
                "features": {},
            }
        by_label[label_id]["features"][feature_name] = feature_value

    ordered_feature_names = sorted(feature_names)
    ordered_samples = [by_label[key] for key in sorted(by_label)]

    x: list[list[float]] = []
    y: list[int] = []
    for sample in ordered_samples:
        x.append([float(sample["features"].get(name, 0.0)) for name in ordered_feature_names])
        y.append(int(sample["target"]))

    return ordered_feature_names, x, y, day_values


def run_clr_training_job(
    conn: sqlite3.Connection,
    *,
    min_labeled_rows: int = 20,
    min_labeled_days: int = 5,
) -> int | None:
    started_at = _utc_now()
    run_cur = conn.execute(
        """
        INSERT INTO clr_training_runs(started_at_utc, status, row_count, day_count, notes)
        VALUES (?, ?, 0, 0, ?)
        """,
        (started_at, "started", ""),
    )
    run_id = int(run_cur.lastrowid)

    rows = conn.execute(
        """
        SELECT label_id, feature_name, feature_value, target, local_date
        FROM vw_clr_training_dataset
        """
    ).fetchall()
    feature_names, x, y, day_values = _build_training_matrix(list(rows))
    row_count = len(x)
    day_count = len(day_values)

    if row_count < min_labeled_rows or day_count < min_labeled_days:
        conn.execute(
            """
            UPDATE clr_training_runs
            SET finished_at_utc = ?, status = ?, row_count = ?, day_count = ?, notes = ?
            WHERE id = ?
            """,
            (_utc_now(), "skipped", row_count, day_count, "training gate not met", run_id),
        )
        conn.commit()
        return None

    if len(set(y)) < 2:
        conn.execute(
            """
            UPDATE clr_training_runs
            SET finished_at_utc = ?, status = ?, row_count = ?, day_count = ?, notes = ?
            WHERE id = ?
            """,
            (_utc_now(), "skipped", row_count, day_count, "need both classes", run_id),
        )
        conn.commit()
        return None

    try:
        logistic_regression_cls = _get_logistic_regression_class()
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        conn.execute(
            """
            UPDATE clr_training_runs
            SET finished_at_utc = ?, status = ?, row_count = ?, day_count = ?, notes = ?
            WHERE id = ?
            """,
            (_utc_now(), "failed", row_count, day_count, f"missing sklearn: {exc}", run_id),
        )
        conn.commit()
        return None

    model = logistic_regression_cls(max_iter=1000, class_weight="balanced")
    model.fit(x, y)
    coefficients = [float(value) for value in model.coef_[0]]
    intercept = float(model.intercept_[0])
    classes = [int(value) for value in model.classes_]
    artifact_json = json.dumps(
        {
            "model": {
                "type": "logistic_regression",
                "coefficients": coefficients,
                "intercept": intercept,
                "classes": classes,
            },
            "feature_names": feature_names,
        },
        sort_keys=True,
    )
    conn.execute(
        """
        INSERT INTO clr_model_artifacts(
            run_id, created_at_utc, model_type, feature_set_version, artifact_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, _utc_now(), "sklearn_logistic_regression", "v1", artifact_json),
    )
    conn.execute(
        """
        UPDATE clr_training_runs
        SET finished_at_utc = ?, status = ?, row_count = ?, day_count = ?, notes = ?
        WHERE id = ?
        """,
        (_utc_now(), "completed", row_count, day_count, "ok", run_id),
    )
    conn.commit()
    return run_id
