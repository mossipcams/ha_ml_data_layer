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
    }
    app.listen_event = lambda callback, event_name: events.append((event_name, callback))
    app.run_daily = lambda callback, schedule_time: schedules.append((callback, schedule_time))
    app.parse_time = lambda value: value
    app.log = lambda _: None

    app.initialize()

    assert len(events) == 1
    assert events[0][0] == "state_changed"
    assert len(schedules) == 1
    assert schedules[0][1] == "03:30:00"
