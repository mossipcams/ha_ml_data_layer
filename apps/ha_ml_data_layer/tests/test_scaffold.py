"""Scaffold checks for the AppDaemon ML data layer package."""

from importlib import import_module
from pathlib import Path


def test_appdaemon_ml_package_scaffold_imports() -> None:
    modules = [
        "appdaemon_ml",
        "appdaemon_ml.app",
        "appdaemon_ml.config",
        "appdaemon_ml.db",
        "appdaemon_ml.ingest",
        "appdaemon_ml.features",
        "appdaemon_ml.labels",
        "appdaemon_ml.lightgbm_train",
        "appdaemon_ml.bocpd_train",
        "appdaemon_ml.contracts",
    ]

    for name in modules:
        assert import_module(name)


def test_hacs_appdaemon_payload_layout() -> None:
    tests_dir = Path(__file__).resolve().parent
    app_root = tests_dir.parent
    repo_root = tests_dir.parents[2]
    apps_root = repo_root / "apps"

    assert (app_root / "ha_ml_data_layer.py").is_file()
    assert (app_root / "appdaemon_ml" / "__init__.py").is_file()
    assert not (apps_root / "__init__.py").exists()
    assert not (repo_root / "__init__.py").exists()


def test_module_path_examples_match_repo_layout() -> None:
    tests_dir = Path(__file__).resolve().parent
    repo_root = tests_dir.parents[2]

    expected = "module: ha_ml_data_layer.ha_ml_data_layer"
    for rel_path in (
        Path("README.md"),
        Path("apps/ha_ml_data_layer/apps.yaml"),
        Path("apps/ha_ml_data_layer/apps.example.yaml"),
    ):
        content = (repo_root / rel_path).read_text(encoding="utf-8")
        assert expected in content
