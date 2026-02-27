from __future__ import annotations

import importlib
import sys
import types


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
