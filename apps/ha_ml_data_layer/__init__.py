"""Package entrypoint for AppDaemon module loading."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "AppDaemonMLDataLayer":
        from .ha_ml_data_layer import AppDaemonMLDataLayer

        return AppDaemonMLDataLayer
    raise AttributeError(name)


__all__ = ["AppDaemonMLDataLayer"]
