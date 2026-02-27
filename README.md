# ha_ml_data_layer

AppDaemon-managed ML data layer for Home Assistant using SQLite contracts.

## HACS layout

This repository is structured for HACS AppDaemon:

- `apps/ha_ml_data_layer/appdaemon_ml/...`

## AppDaemon config example

Example `apps.yaml` entry:

```yaml
ha_ml_data_layer:
  module: ha_ml_data_layer
  class: AppDaemonMLDataLayer
  db_path: /addon_configs/a0d7b954_appdaemon/appdaemon/ha_ml_data_layer.db
  nightly_time: "03:00:00"
  retention_time: "04:00:00"
  raw_retention_days: 30
  feature_retention_days: 90
  timezone_name: UTC
```
