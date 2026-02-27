from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from appdaemon_ml.app import AppDaemonMLDataLayer
from appdaemon_ml.db import connect, ensure_schema
from appdaemon_ml.ingest import record_raw_event


def test_record_raw_event_inserts_and_dedupes(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        occurred = datetime(2026, 2, 25, 6, 0, tzinfo=UTC)
        row_id_1 = record_raw_event(
            conn,
            event_type="state_changed",
            entity_id="sensor.bedroom",
            state="on",
            occurred_at=occurred,
        )
        row_id_2 = record_raw_event(
            conn,
            event_type="state_changed",
            entity_id="sensor.bedroom",
            state="on",
            occurred_at=occurred,
        )
        assert row_id_1 is not None
        assert row_id_2 is None

        rows = conn.execute(
            "SELECT occurred_at_utc FROM raw_events ORDER BY id ASC"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0].endswith("+00:00")
    finally:
        conn.close()


def test_core_handle_event_respects_ingestion_rules(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    app = AppDaemonMLDataLayer(db_path=db_path, timezone_name="UTC")
    app.initialize()

    occurred = datetime(2026, 2, 25, 6, 0, tzinfo=UTC)
    blocked_row = app.handle_event(
        event_type="state_changed",
        entity_id="sensor.bedroom",
        state="on",
        occurred_at=occurred,
    )
    assert blocked_row is None

    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO ingestion_rules(entity_id, state, source, updated_at_utc)
            VALUES (?, ?, ?, ?)
            """,
            ("sensor.bedroom", "on", "mindml:test", "2026-02-25T06:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    allowed_row = app.handle_event(
        event_type="state_changed",
        entity_id="sensor.bedroom",
        state="on",
        occurred_at=occurred,
    )
    assert allowed_row is not None

    conn = connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_core_handle_event_sets_last_ingest_only_on_insert(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    app = AppDaemonMLDataLayer(db_path=db_path, timezone_name="UTC")
    app.initialize()

    occurred = datetime(2026, 2, 25, 6, 0, tzinfo=UTC)
    app.handle_event(
        event_type="state_changed",
        entity_id="sensor.bedroom",
        state="on",
        occurred_at=occurred,
    )

    conn = connect(db_path)
    try:
        blocked_meta = conn.execute(
            "SELECT value FROM metadata WHERE key = 'last_ingest_at'"
        ).fetchone()
        assert blocked_meta is None
        conn.execute(
            """
            INSERT INTO ingestion_rules(entity_id, state, source, updated_at_utc)
            VALUES (?, ?, ?, ?)
            """,
            ("sensor.bedroom", "on", "mindml:test", "2026-02-25T06:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    app.handle_event(
        event_type="state_changed",
        entity_id="sensor.bedroom",
        state="on",
        occurred_at=occurred,
    )

    conn = connect(db_path)
    try:
        allowed_meta = conn.execute(
            "SELECT value FROM metadata WHERE key = 'last_ingest_at'"
        ).fetchone()
        assert allowed_meta is not None
    finally:
        conn.close()
