"""Root package shim for AppDaemon import compatibility."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "AppDaemonMLDataLayer":
        from .apps.ha_ml_data_layer.ha_ml_data_layer import AppDaemonMLDataLayer

        return AppDaemonMLDataLayer
    raise AttributeError(name)


__all__ = ["AppDaemonMLDataLayer"]
