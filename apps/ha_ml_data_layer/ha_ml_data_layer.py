"""HACS/AppDaemon entrypoint for the ha_ml_data_layer app."""

from __future__ import annotations

from pathlib import Path

import appdaemon.plugins.hass.hassapi as hass

from .appdaemon_ml.app import AppDaemonMLDataLayer as CoreDataLayer


class AppDaemonMLDataLayer(hass.Hass):
    """AppDaemon-compatible wrapper around the core data-layer class."""

    def initialize(self) -> None:
        db_path = Path(self.args.get("db_path", "/config/appdaemon/ha_ml_data_layer.db"))
        timezone_name = self.args.get("timezone_name", "UTC")
        self._core = CoreDataLayer(db_path=db_path, timezone_name=timezone_name)
        self._core.initialize()
        self.log(f"ha_ml_data_layer initialized with db_path={db_path}")


__all__ = ["AppDaemonMLDataLayer"]
