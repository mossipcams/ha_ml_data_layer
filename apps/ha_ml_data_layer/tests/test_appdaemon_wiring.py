from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime


def _install_fake_appdaemon_hass() -> None:
    appdaemon_pkg = types.ModuleType("appdaemon")
    plugins_pkg = types.ModuleType("appdaemon.plugins")
    hass_pkg = types.ModuleType("appdaemon.plugins.hass")
    hassapi_mod = types.ModuleType("appdaemon.plugins.hass.hassapi")

    class FakeHass:
        pass

    hassapi_mod.Hass = FakeHass
    sys.modules["appdaemon"] = appdaemon_pkg
    sys.modules["appdaemon.plugins"] = plugins_pkg
    sys.modules["appdaemon.plugins.hass"] = hass_pkg
    sys.modules["appdaemon.plugins.hass.hassapi"] = hassapi_mod


def test_initialize_registers_event_listener_and_nightly_schedule(monkeypatch) -> None:
    _install_fake_appdaemon_hass()
    module = importlib.import_module("ha_ml_data_layer")

    events: list[tuple[str, object]] = []
    schedules: list[tuple[object, object]] = []
    app = module.AppDaemonMLDataLayer.__new__(module.AppDaemonMLDataLayer)
    app.args = {
        "db_path": "/tmp/ha_ml_data_layer.db",
        "timezone_name": "UTC",
        "event_name": "state_changed",
        "nightly_time": "03:30:00",
        "retention_time": "04:00:00",
    }
    app.listen_event = lambda callback, event_name: events.append((event_name, callback))
    app.run_daily = lambda callback, schedule_time: schedules.append((callback, schedule_time))
    app.parse_time = lambda value: value
    app.log = lambda _: None

    app.initialize()

    assert len(events) == 1
    assert events[0][0] == "state_changed"
    assert len(schedules) == 2
    assert schedules[0][1] == "03:30:00"
    assert schedules[1][1] == "04:00:00"


def test_on_ha_event_forwards_entity_and_state_to_core() -> None:
    _install_fake_appdaemon_hass()
    module = importlib.import_module("ha_ml_data_layer")
    app = module.AppDaemonMLDataLayer.__new__(module.AppDaemonMLDataLayer)

    calls: list[dict] = []

    class FakeCore:
        def handle_event(self, **kwargs):
            calls.append(kwargs)

    app._core = FakeCore()

    app._on_ha_event(
        "state_changed",
        {"entity_id": "sensor.anything", "new_state": {"state": "on"}},
        {},
    )
    assert len(calls) == 1
    assert calls[0]["entity_id"] == "sensor.anything"
    assert calls[0]["state"] == "on"
    assert calls[0]["attributes"]["entity_id"] == "sensor.anything"


def test_retention_callback_uses_configured_days() -> None:
    _install_fake_appdaemon_hass()
    module = importlib.import_module("ha_ml_data_layer")
    app = module.AppDaemonMLDataLayer.__new__(module.AppDaemonMLDataLayer)
    calls: list[tuple[int, int]] = []

    class FakeCore:
        def run_retention(self, *, raw_days: int, feature_days: int) -> None:
            calls.append((raw_days, feature_days))

    app._core = FakeCore()
    app._raw_retention_days = 10
    app._feature_retention_days = 45
    app.log = lambda _: None

    app._run_retention({})

    assert calls == [(10, 45)]


def test_nightly_pipeline_uses_previous_local_date_before_sleep_end() -> None:
    _install_fake_appdaemon_hass()
    module = importlib.import_module("ha_ml_data_layer")
    app = module.AppDaemonMLDataLayer.__new__(module.AppDaemonMLDataLayer)

    calls: list[dict] = []

    class FakeCore:
        def run_nightly_pipeline(self, **kwargs):
            calls.append(kwargs)

    app._core = FakeCore()
    app._feature_window_hours = 24
    app._sleep_start_entity = "input_datetime.sleep_start"
    app._sleep_end_entity = "input_datetime.sleep_end"
    app.get_state = lambda entity: "23:00:00" if "start" in entity else "07:00:00"
    app.datetime = lambda: datetime(2026, 2, 27, 3, 0, 0)

    app._run_nightly_pipeline({})

    assert len(calls) == 1
    assert calls[0]["local_date"] == "2026-02-26"


def test_initialize_triggers_startup_retrain() -> None:
    _install_fake_appdaemon_hass()
    module = importlib.import_module("ha_ml_data_layer")

    calls: list[str] = []

    class FakeCore:
        def __init__(self, *, db_path, timezone_name):
            self.db_path = db_path
            self.timezone_name = timezone_name

        def initialize(self) -> None:
            calls.append("initialize")

        def run_nightly_pipeline(self, **kwargs) -> None:
            calls.append("run_nightly_pipeline")

        def run_startup_retrain(self) -> None:
            calls.append("run_startup_retrain")

    module.CoreDataLayer = FakeCore

    app = module.AppDaemonMLDataLayer.__new__(module.AppDaemonMLDataLayer)
    app.args = {
        "db_path": "/tmp/ha_ml_data_layer.db",
        "timezone_name": "UTC",
        "event_name": "state_changed",
        "nightly_time": "03:30:00",
        "retention_time": "04:00:00",
    }
    app.listen_event = lambda callback, event_name: None
    app.run_daily = lambda callback, schedule_time: None
    app.parse_time = lambda value: value
    app.datetime = lambda: datetime(2026, 2, 27, 3, 0, 0)
    app.get_state = lambda entity: "23:00:00" if "start" in entity else "07:00:00"
    app.log = lambda _: None

    app.initialize()

    assert calls == ["initialize", "run_nightly_pipeline", "run_startup_retrain"]
