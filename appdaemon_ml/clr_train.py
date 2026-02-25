"""CLR training job routines."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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
        SELECT feature_name, feature_value, target, local_date
        FROM vw_clr_training_dataset
        """
    ).fetchall()
    row_count = len(rows)
    day_count = len({row["local_date"] for row in rows})

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

    positive = defaultdict(list)
    negative = defaultdict(list)
    for row in rows:
        if row["target"] == 1:
            positive[row["feature_name"]].append(float(row["feature_value"]))
        else:
            negative[row["feature_name"]].append(float(row["feature_value"]))

    weights: dict[str, float] = {}
    for name in set(positive) | set(negative):
        pos_mean = sum(positive[name]) / len(positive[name]) if positive[name] else 0.0
        neg_mean = sum(negative[name]) / len(negative[name]) if negative[name] else 0.0
        weights[name] = pos_mean - neg_mean

    artifact_json = json.dumps({"weights": weights}, sort_keys=True)
    conn.execute(
        """
        INSERT INTO clr_model_artifacts(
            run_id, created_at_utc, model_type, feature_set_version, artifact_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, _utc_now(), "logistic_regression_like", "v1", artifact_json),
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
