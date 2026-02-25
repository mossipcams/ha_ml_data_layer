"""Root package shim for AppDaemon import compatibility."""

from .apps.ha_ml_data_layer.ha_ml_data_layer import AppDaemonMLDataLayer

__all__ = ["AppDaemonMLDataLayer"]
