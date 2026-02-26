"""HACS/AppDaemon entrypoint for the ha_ml_data_layer app."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import appdaemon.plugins.hass.hassapi as hass

from appdaemon_ml.app import AppDaemonMLDataLayer as CoreDataLayer

DEFAULT_IMPORTANT_OBSERVATIONS: dict[str, set[str]] = {
    "binary_sensor.bedtime": {"on"},
    "sensor.matts_iphone_ble_area": {"Bedroom", "Living Room"},
    "device_tracker.mattiphone": {"home"},
    "binary_sensor.door_sensor_group": {"off"},
    "input_boolean.thermostat_motion_toggle": {"off"},
    "input_boolean.almost_bedtime": {"on"},
    "binary_sensor.apollo_msr_2_173d50_radar_target": {"on"},
}


class AppDaemonMLDataLayer(hass.Hass):
    """AppDaemon-compatible wrapper around the core data-layer class."""

    def initialize(self) -> None:
        db_path = Path(self.args.get("db_path", "/config/appdaemon/ha_ml_data_layer.db"))
        self._timezone_name = self.args.get("timezone_name", "UTC")
        self._event_name = self.args.get("event_name", "state_changed")
        self._nightly_time = self.args.get("nightly_time", "03:00:00")
        self._sleep_start_entity = self.args.get(
            "sleep_start_entity", "input_datetime.sleep_start"
        )
        self._sleep_end_entity = self.args.get(
            "sleep_end_entity", "input_datetime.sleep_end"
        )
        self._feature_window_hours = int(self.args.get("feature_window_hours", 24))
        self._retention_time = self.args.get("retention_time", "04:00:00")
        self._raw_retention_days = int(self.args.get("raw_retention_days", 30))
        self._feature_retention_days = int(self.args.get("feature_retention_days", 90))
        self._important_observations = DEFAULT_IMPORTANT_OBSERVATIONS

        timezone_name = self._timezone_name
        self._core = CoreDataLayer(db_path=db_path, timezone_name=timezone_name)
        self._core.initialize()
        self.listen_event(self._on_ha_event, self._event_name)
        self.run_daily(self._run_nightly_pipeline, self.parse_time(self._nightly_time))
        self.run_daily(self._run_retention, self.parse_time(self._retention_time))
        self.log(f"ha_ml_data_layer initialized with db_path={db_path}")

    def _on_ha_event(self, event_name: str, data: dict, kwargs: dict) -> None:
        entity_id = data.get("entity_id")
        new_state = data.get("new_state")
        state = new_state.get("state") if isinstance(new_state, dict) else None
        if not entity_id or state is None:
            return

        allowed_states = self._important_observations.get(entity_id)
        if not allowed_states:
            return

        state_str = str(state)
        is_match = state_str in allowed_states
        if not is_match:
            return

        attributes: dict[str, Any] = {
            "important_observation": True,
            "entity_id": entity_id,
            "state": state_str,
        }
        self._core.handle_event(
            event_type=event_name,
            entity_id=entity_id,
            state=state_str,
            attributes=attributes,
        )

    def _run_nightly_pipeline(self, kwargs: dict) -> None:
        now_local = self.datetime()
        window_end = now_local
        window_start = now_local - timedelta(hours=self._feature_window_hours)
        sleep_start = str(self.get_state(self._sleep_start_entity) or "23:00:00")
        sleep_end = str(self.get_state(self._sleep_end_entity) or "07:00:00")
        local_date = now_local.date().isoformat()
        self._core.run_nightly_pipeline(
            local_date=local_date,
            sleep_start=sleep_start,
            sleep_end=sleep_end,
            window_start=window_start,
            window_end=window_end,
        )

    def _run_retention(self, kwargs: dict) -> None:
        self._core.run_retention(
            raw_days=self._raw_retention_days,
            feature_days=self._feature_retention_days,
        )
        self.log(
            "ha_ml_data_layer retention completed "
            f"(raw_days={self._raw_retention_days}, feature_days={self._feature_retention_days})"
        )


__all__ = ["AppDaemonMLDataLayer"]
