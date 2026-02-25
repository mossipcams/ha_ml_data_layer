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
  db_path: /config/appdaemon/ha_ml_data_layer.db
  timezone_name: UTC
```
