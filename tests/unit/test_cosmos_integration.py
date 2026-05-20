"""Tests for dbt manifest ↔ drt sync mapping and Cosmos integration.

Tests ``drt.integrations.dbt.map_syncs_to_manifest`` — the core function
that maps dbt model names to drt SyncConfig instances by scanning the
``syncs/`` directory and matching ``ref('model_name')`` references against
a parsed dbt manifest.

Tests ``drt.integrations.cosmos.attach_drt_to_cosmos`` with mocking —
no live Airflow or Cosmos environment required.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build test fixtures on tmp_path
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, nodes: dict[str, dict]) -> Path:
    """Write a dbt manifest.json into tmp_path/target/manifest.json."""
    manifest_dir = tmp_path / "target"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"nodes": nodes}))
    return manifest_path


def _write_sync_yaml(
    syncs_dir: Path, filename: str, name: str, model: str
) -> Path:
    """Write a minimal sync YAML file into the syncs/ directory.

    Uses a simple REST API destination as a placeholder.
    """
    syncs_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "model": model,
        "destination": {
            "type": "rest_api",
            "url": "https://example.com/api",
            "method": "POST",
            "auth": {"type": "bearer", "token_env": "DUMMY_TOKEN"},
            "body_template": '{"dummy": true}',
        },
        "sync": {
            "mode": "full",
        },
    }
    path = syncs_dir / filename
    path.write_text(yaml.dump(data))
    return path


def _make_model_node(name: str, relation_name: str) -> dict:
    """Build a dbt manifest model node dict."""
    return {
        "name": name,
        "resource_type": "model",
        "relation_name": relation_name,
        "description": f"Model: {name}",
        "schema": "public",
    }


def _make_seed_node(name: str) -> dict:
    """Build a dbt manifest seed node dict."""
    return {
        "name": name,
        "resource_type": "seed",
        "relation_name": f'"analytics"."public"."{name}"',
    }


def _make_snapshot_node(name: str) -> dict:
    """Build a dbt manifest snapshot node dict."""
    return {
        "name": name,
        "resource_type": "snapshot",
        "relation_name": f'"analytics"."snapshots"."{name}"',
    }


def _write_bad_sync_yaml(syncs_dir: Path, filename: str) -> Path:
    """Write a sync YAML file that parses but fails SyncConfig validation.

    Writes valid YAML with only a ``name`` field — missing ``model`` and
    ``destination``, which will trigger a ``ValidationError`` inside
    ``load_syncs_safe``. The file ends up in ``SyncLoadResult.errors``
    rather than ``.syncs``.
    """
    syncs_dir.mkdir(parents=True, exist_ok=True)
    path = syncs_dir / filename
    path.write_text(yaml.dump({"name": "bad_sync"}))
    return path


# ---------------------------------------------------------------------------
# map_syncs_to_manifest tests
# ---------------------------------------------------------------------------


class TestMapSyncsToManifest:
    """Tests for ``drt.integrations.dbt.map_syncs_to_manifest``."""

    def test_model_in_manifest_and_sync_ref_matches(
        self, tmp_path: Path
    ) -> None:
        """Manifest has dim_users, drt sync has ref('dim_users') → included."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.dim_users": _make_model_node(
                "dim_users", '"analytics"."public"."dim_users"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(syncs_dir, "dim_users.yml", "dim_users_load", "ref('dim_users')")

        # map_syncs_to_manifest accepts the manifest dict + project dir
        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        assert "dim_users" in result
        assert len(result["dim_users"]) == 1
        assert result["dim_users"][0].name == "dim_users_load"

    def test_one_model_multiple_syncs(
        self, tmp_path: Path
    ) -> None:
        """One model referenced by two different sync YAML files → both mapped."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.fct_orders": _make_model_node(
                "fct_orders", '"analytics"."public"."fct_orders"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(syncs_dir, "orders_to_hubspot.yml", "orders_hubspot", "ref('fct_orders')")
        _write_sync_yaml(syncs_dir, "orders_to_sheets.yml", "orders_sheets", "ref('fct_orders')")

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        assert "fct_orders" in result
        assert len(result["fct_orders"]) == 2
        sync_names = {s.name for s in result["fct_orders"]}
        assert sync_names == {"orders_hubspot", "orders_sheets"}

    def test_raw_sql_not_included(self, tmp_path: Path) -> None:
        """Sync with raw SQL (not ref()) → not in mapping."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.my_table": _make_model_node(
                "my_table", '"analytics"."public"."my_table"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(
            syncs_dir,
            "raw_sql_sync.yml",
            "raw_sync",
            "SELECT * FROM orders WHERE status = 'active'",
        )

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        # Raw SQL syncs should not appear in the mapping
        assert result == {} or "my_table" not in result
        # No model key should have the raw sync
        all_syncs = []
        for syncs in result.values():
            all_syncs.extend(syncs)
        assert not any(s.name == "raw_sync" for s in all_syncs)

    def test_ref_to_nonexistent_model_not_included(
        self, tmp_path: Path
    ) -> None:
        """Sync with ref('nonexistent_model') where model not in manifest → excluded."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.existing_model": _make_model_node(
                "existing_model", '"analytics"."public"."existing_model"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(syncs_dir, "orphan_sync.yml", "orphan_load", "ref('nonexistent_model')")

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        assert "nonexistent_model" not in result
        assert "existing_model" not in result  # no sync references it

    def test_empty_manifest_returns_empty_mapping(
        self, tmp_path: Path
    ) -> None:
        """Empty manifest → empty mapping, even if syncs exist."""
        from drt.integrations.dbt import map_syncs_to_manifest

        _write_manifest(tmp_path, {})

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(syncs_dir, "test_sync.yml", "test_load", "ref('some_model')")

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        assert result == {}

    def test_no_syncs_directory_returns_empty_mapping(
        self, tmp_path: Path
    ) -> None:
        """No syncs/ directory → empty mapping."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.my_model": _make_model_node(
                "my_model", '"analytics"."public"."my_model"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        # Don't create syncs/ directory
        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        assert result == {}

    def test_only_models_matched_not_seeds_or_snapshots(
        self, tmp_path: Path
    ) -> None:
        """Seeds and snapshots in manifest → only models are matched."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.dim_customers": _make_model_node(
                "dim_customers", '"analytics"."public"."dim_customers"'
            ),
            "seed.my_project.country_codes": _make_seed_node("country_codes"),
            "snapshot.my_project.dim_customers_snapshot": _make_snapshot_node(
                "dim_customers"
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(
            syncs_dir, "customers_sync.yml", "customers_load", "ref('dim_customers')"
        )
        _write_sync_yaml(
            syncs_dir, "country_seed_sync.yml", "country_load", "ref('country_codes')"
        )

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        # dim_customers is a model → should match
        assert "dim_customers" in result
        assert len(result["dim_customers"]) == 1
        assert result["dim_customers"][0].name == "customers_load"

        # country_codes is a seed → should NOT match
        assert "country_codes" not in result

    def test_multiple_models_each_with_syncs(
        self, tmp_path: Path
    ) -> None:
        """Multiple models each referenced by syncs → all properly mapped."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.dim_users": _make_model_node(
                "dim_users", '"analytics"."public"."dim_users"'
            ),
            "model.my_project.fct_orders": _make_model_node(
                "fct_orders", '"analytics"."public"."fct_orders"'
            ),
            "model.my_project.agg_daily": _make_model_node(
                "agg_daily", '"analytics"."public"."agg_daily"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(syncs_dir, "users.yml", "users_hubspot", "ref('dim_users')")
        _write_sync_yaml(syncs_dir, "orders.yml", "orders_slack", "ref('fct_orders')")
        # agg_daily has no sync referencing it

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        assert "dim_users" in result
        assert "fct_orders" in result
        assert "agg_daily" not in result  # no sync references it

    def test_ref_with_double_quotes(
        self, tmp_path: Path
    ) -> None:
        """ref(\"model_name\") should also be detected."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.products": _make_model_node(
                "products", '"analytics"."public"."products"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(syncs_dir, "products_sync.yml", "products_load", 'ref("products")')

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        assert "products" in result
        assert len(result["products"]) == 1

    def test_model_name_case_sensitive_match(
        self, tmp_path: Path
    ) -> None:
        """Model name matching is case-sensitive."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.MyModel": _make_model_node(
                "MyModel", '"analytics"."public"."MyModel"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(syncs_dir, "case_test.yml", "case_load", "ref('MyModel')")
        _write_sync_yaml(syncs_dir, "case_test2.yml", "case_load2", "ref('mymodel')")

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        # Exact match should work
        assert "MyModel" in result
        assert len(result["MyModel"]) == 1
        # Case-mismatched ref should not match
        assert "mymodel" not in result

    def test_manifest_with_non_node_keys_ignored(
        self, tmp_path: Path
    ) -> None:
        """Extra top-level manifest keys (metadata, etc.) are ignored."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.my_model": _make_model_node(
                "my_model", '"analytics"."public"."my_model"'
            ),
        }
        manifest_path = _write_manifest(tmp_path, nodes)

        # Add extra keys to the manifest
        manifest_data = json.loads(manifest_path.read_text())
        manifest_data["metadata"] = {"dbt_version": "1.8.0"}
        manifest_data["docs"] = {"some_doc": "value"}

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(syncs_dir, "test_sync.yml", "test_load", "ref('my_model')")

        result = map_syncs_to_manifest(manifest_data, tmp_path)

        assert "my_model" in result
        assert len(result["my_model"]) == 1

    def test_bad_yaml_file_skipped_valid_syncs_still_work(
        self, tmp_path: Path
    ) -> None:
        """One valid sync + one invalid YAML → valid sync still mapped, bad one skipped."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.dim_users": _make_model_node(
                "dim_users", '"analytics"."public"."dim_users"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        _write_sync_yaml(syncs_dir, "valid.yml", "valid_load", "ref('dim_users')")
        _write_bad_sync_yaml(syncs_dir, "bad.yml")

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        assert "dim_users" in result
        assert len(result["dim_users"]) == 1
        assert result["dim_users"][0].name == "valid_load"

    def test_all_bad_yaml_returns_empty_mapping(
        self, tmp_path: Path
    ) -> None:
        """All sync YAML files are malformed → empty mapping, no crash."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.my_model": _make_model_node(
                "my_model", '"analytics"."public"."my_model"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        _write_bad_sync_yaml(syncs_dir, "bad1.yml")
        _write_bad_sync_yaml(syncs_dir, "bad2.yml")

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        assert result == {}

    def test_bad_yaml_with_valid_ref_ignored(
        self, tmp_path: Path
    ) -> None:
        """Malformed YAML containing ref('model') text → NOT matched by map_syncs_to_manifest."""
        from drt.integrations.dbt import map_syncs_to_manifest

        nodes = {
            "model.my_project.my_model": _make_model_node(
                "my_model", '"analytics"."public"."my_model"'
            ),
        }
        _write_manifest(tmp_path, nodes)

        syncs_dir = tmp_path / "syncs"
        syncs_dir.mkdir(parents=True, exist_ok=True)
        path = syncs_dir / "has_ref_but_invalid.yml"
        # YAML contains ref('my_model') text but is missing required fields
        # → fails SyncConfig.model_validate → lands in errors, not syncs
        path.write_text(yaml.dump({"name": "bad", "model": "ref('my_model')"}))

        manifest_data = json.loads((tmp_path / "target" / "manifest.json").read_text())
        result = map_syncs_to_manifest(manifest_data, tmp_path)

        # The ref('my_model') text appeared in a file that was never parsed
        # into a SyncConfig, so it should NOT appear in the mapping.
        assert "my_model" not in result


# ---------------------------------------------------------------------------
# _resolve_manifest and _read_manifest_from_project_config tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cosmos():
    """Inject a mock cosmos.config.ProjectConfig into sys.modules."""
    mock_pc = type("ProjectConfig", (), {})
    mock_module = MagicMock(ProjectConfig=mock_pc)
    sys.modules["cosmos"] = MagicMock(config=mock_module)
    sys.modules["cosmos.config"] = mock_module
    yield mock_pc
    sys.modules.pop("cosmos", None)
    sys.modules.pop("cosmos.config", None)


class TestResolveManifest:
    """Tests for ``drt.integrations.cosmos._resolve_manifest``."""

    def test_manifest_dict_passed_returned_as_is(self):
        """When manifest dict is passed, it's returned as-is (no file I/O)."""
        from drt.integrations.cosmos import _resolve_manifest

        manifest_data = {"nodes": {"model.test.foo": {"name": "foo"}}}
        result = _resolve_manifest(
            manifest=manifest_data,
            project_config=None,
            drt_project_dir=Path("/tmp"),
        )
        assert result is manifest_data
        assert result == manifest_data

    def test_project_config_passed_delegates_to_read(self):
        """When project_config is passed (no manifest), delegates to _read_manifest_from_project_config."""
        from drt.integrations.cosmos import _resolve_manifest

        expected = {"nodes": {}}
        with patch(
            "drt.integrations.cosmos._read_manifest_from_project_config",
            return_value=expected,
        ) as mock_read:
            result = _resolve_manifest(
                manifest=None,
                project_config="fake_config",
                drt_project_dir=Path("/tmp"),
            )
            mock_read.assert_called_once_with("fake_config")
            assert result is expected

    def test_neither_passed_reads_from_disk(self, tmp_path: Path):
        """When neither manifest nor project_config is passed, reads from disk."""
        from drt.integrations.cosmos import _resolve_manifest

        nodes = {
            "model.test.foo": _make_model_node("foo", '"analytics"."public"."foo"'),
        }
        _write_manifest(tmp_path, nodes)
        expected = json.loads((tmp_path / "target" / "manifest.json").read_text())

        result = _resolve_manifest(
            manifest=None,
            project_config=None,
            drt_project_dir=tmp_path,
        )
        assert result == expected

    def test_neither_passed_file_missing_raises(self, tmp_path: Path):
        """When neither is passed and manifest.json is missing, raises FileNotFoundError."""
        from drt.integrations.cosmos import _resolve_manifest

        with pytest.raises(FileNotFoundError, match="manifest.json not found"):
            _resolve_manifest(
                manifest=None,
                project_config=None,
                drt_project_dir=tmp_path,
            )

    def test_both_manifest_and_project_config_logs_warning(self):
        """When both manifest and project_config are passed, logs a warning."""
        from drt.integrations.cosmos import _resolve_manifest

        manifest_data = {"nodes": {"model.test.foo": {"name": "foo"}}}
        with patch("drt.integrations.cosmos.logger") as mock_logger:
            result = _resolve_manifest(
                manifest=manifest_data,
                project_config=MagicMock(),
                drt_project_dir=Path("/tmp"),
            )
            mock_logger.warning.assert_called_once()
            assert result is manifest_data

    def test_malformed_manifest_json_raises_value_error(self, tmp_path: Path):
        """Manifest from disk contains malformed JSON → ValueError."""
        from drt.integrations.cosmos import _resolve_manifest

        # Write malformed JSON to the expected location
        target_dir = tmp_path / "target"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "manifest.json").write_text("{broken")

        with pytest.raises(ValueError, match="not valid JSON"):
            _resolve_manifest(
                manifest=None,
                project_config=None,
                drt_project_dir=tmp_path,
            )


class TestReadManifestFromProjectConfig:
    """Tests for ``drt.integrations.cosmos._read_manifest_from_project_config``."""

    def test_valid_local_manifest_path(self, tmp_path: Path, mock_cosmos):
        """Valid local manifest_path → reads and returns dict."""
        from drt.integrations.cosmos import _read_manifest_from_project_config

        manifest_data = {"nodes": {"model.test.foo": {"name": "foo"}}}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        config = mock_cosmos()
        config.manifest_path = str(manifest_path)

        result = _read_manifest_from_project_config(config)
        assert result == manifest_data

    def test_manifest_path_is_none_raises(self, mock_cosmos):
        """manifest_path is None → raises FileNotFoundError."""
        from drt.integrations.cosmos import _read_manifest_from_project_config

        config = mock_cosmos()
        config.manifest_path = None

        with pytest.raises(FileNotFoundError, match="manifest_path is None"):
            _read_manifest_from_project_config(config)

    def test_manifest_path_is_remote_uri_raises(self, mock_cosmos):
        """manifest_path is a remote URI (s3://...) → raises ValueError."""
        from drt.integrations.cosmos import _read_manifest_from_project_config

        config = mock_cosmos()
        config.manifest_path = "s3://bucket/manifest.json"

        with pytest.raises(ValueError, match="remote URI"):
            _read_manifest_from_project_config(config)

    def test_manifest_path_nonexistent_file_raises(
        self, tmp_path: Path, mock_cosmos
    ):
        """manifest_path points to nonexistent file → raises FileNotFoundError."""
        from drt.integrations.cosmos import _read_manifest_from_project_config

        config = mock_cosmos()
        config.manifest_path = str(tmp_path / "nonexistent.json")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            _read_manifest_from_project_config(config)

    def test_config_not_projectconfig_instance_raises(self, mock_cosmos):
        """config is not a ProjectConfig instance → raises TypeError."""
        from drt.integrations.cosmos import _read_manifest_from_project_config

        with pytest.raises(TypeError, match="must be a cosmos.config.ProjectConfig"):
            _read_manifest_from_project_config({"not": "a_project_config"})

    def test_file_uri_allowed(self, tmp_path: Path, mock_cosmos):
        """file:// URI is allowed through the remote-URI check.

        ``file://`` URIs do NOT raise ValueError("remote URI"). However,
        ``Path("file:///...")`` constructs a path that won't exist, so
        a FileNotFoundError is expected instead — confirming the
        remote-URI guard was bypassed.
        """
        from drt.integrations.cosmos import _read_manifest_from_project_config

        config = mock_cosmos()
        config.manifest_path = "file:///nonexistent/manifest.json"

        # Should NOT raise ValueError("remote URI"); may raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            _read_manifest_from_project_config(config)

    def test_cosmos_not_installed_raises_import_error(self):
        """cosmos not installed → ImportError if the import fails."""
        from drt.integrations.cosmos import _read_manifest_from_project_config

        # Ensure cosmos is not in sys.modules
        sys.modules.pop("cosmos", None)
        sys.modules.pop("cosmos.config", None)

        with pytest.raises(ImportError, match="project_config requires astronomer-cosmos"):
            _read_manifest_from_project_config(MagicMock())

    def test_malformed_manifest_json_raises_value_error(
        self, tmp_path: Path, mock_cosmos
    ):
        """Manifest from project_config contains malformed JSON → ValueError."""
        from drt.integrations.cosmos import _read_manifest_from_project_config

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{malformed: yes}")

        config = mock_cosmos()
        config.manifest_path = str(manifest_path)

        with pytest.raises(ValueError, match="not valid JSON"):
            _read_manifest_from_project_config(config)


# ---------------------------------------------------------------------------
# _resolve_upstream_task tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_airflow_taskgroup():
    """Inject a mock airflow.utils.task_group.TaskGroup into sys.modules."""
    mock_tg = type("TaskGroup", (), {})
    mock_module = MagicMock(TaskGroup=mock_tg)
    sys.modules["airflow"] = MagicMock()
    sys.modules["airflow.utils"] = MagicMock()
    sys.modules["airflow.utils.task_group"] = mock_module
    yield mock_tg
    sys.modules.pop("airflow", None)
    sys.modules.pop("airflow.utils", None)
    sys.modules.pop("airflow.utils.task_group", None)


class TestResolveUpstreamTask:
    """Tests for ``drt.integrations.cosmos._resolve_upstream_task``."""

    def test_after_tests_false_returns_task_as_is(self):
        """after_tests=False → returns the same task object unchanged."""
        from drt.integrations.cosmos import _resolve_upstream_task

        task = MagicMock(name="model_task")
        result = _resolve_upstream_task(task, after_tests=False)

        assert result is task

    def test_after_tests_true_plain_task_returns_task(
        self, mock_airflow_taskgroup,
    ):
        """after_tests=True with a plain task (not a TaskGroup) → returns as-is."""
        from drt.integrations.cosmos import _resolve_upstream_task

        task = MagicMock(name="plain_model_task")
        result = _resolve_upstream_task(task, after_tests=True)

        # isinstance(task, TaskGroup) is False → falls through to return task
        assert result is task

    def test_after_tests_true_taskgroup_with_test_child_returns_test(
        self, mock_airflow_taskgroup,
    ):
        """after_tests=True, TaskGroup has a 'test' child → returns the test task."""
        from drt.integrations.cosmos import _resolve_upstream_task

        tg = mock_airflow_taskgroup()
        test_task = MagicMock(name="test_task")
        tg.get_child_by_label = MagicMock(return_value=test_task)

        result = _resolve_upstream_task(tg, after_tests=True)

        assert result is test_task
        tg.get_child_by_label.assert_called_once_with("test")

    def test_after_tests_true_taskgroup_without_test_child_returns_group(
        self, mock_airflow_taskgroup,
    ):
        """after_tests=True, TaskGroup has no 'test' child → returns the TaskGroup."""
        from drt.integrations.cosmos import _resolve_upstream_task

        tg = mock_airflow_taskgroup()
        tg.get_child_by_label = MagicMock(return_value=None)

        result = _resolve_upstream_task(tg, after_tests=True)

        assert result is tg
        tg.get_child_by_label.assert_called_once_with("test")

    def test_after_tests_true_airflow_not_installed_returns_task(self):
        """after_tests=True but Airflow not installed → ImportError caught, returns task."""
        from drt.integrations.cosmos import _resolve_upstream_task

        # Ensure airflow is not in sys.modules
        sys.modules.pop("airflow", None)
        sys.modules.pop("airflow.utils", None)
        sys.modules.pop("airflow.utils.task_group", None)

        task = MagicMock(name="model_task")
        result = _resolve_upstream_task(task, after_tests=True)

        # ImportError caught internally → falls through to return task as-is
        assert result is task


# ---------------------------------------------------------------------------
# __all__ export tests
# ---------------------------------------------------------------------------


class TestCosmosModuleExports:
    """Tests for ``drt.integrations.cosmos`` module-level exports."""

    def test_all_exported(self):
        """__all__ should exist and contain 'attach_drt_to_cosmos'."""
        from drt.integrations import cosmos

        assert hasattr(cosmos, "__all__")
        assert "attach_drt_to_cosmos" in cosmos.__all__


# ---------------------------------------------------------------------------
# attach_drt_to_cosmos tests (fully mocked — no Airflow/Cosmos required)
# ---------------------------------------------------------------------------


class TestAttachDrtToCosmos:
    """Tests for ``drt.integrations.cosmos.attach_drt_to_cosmos``."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_fake_task_group(nodes_dict=None, tasks_dict=None):
        """Create a mock Cosmos DbtTaskGroup with tasks_map and dbt_graph."""
        tg = MagicMock()
        tg.tasks_map = tasks_dict if tasks_dict is not None else {}
        # filtered_nodes.items() must return (unique_id, node) tuples
        if nodes_dict:
            tg.dbt_graph.filtered_nodes.items.return_value = list(
                nodes_dict.items()
            )
        else:
            tg.dbt_graph.filtered_nodes.items.return_value = []
        return tg

    @staticmethod
    def _make_sync_config(name):
        """Create a mock SyncConfig with a name."""
        sync = MagicMock()
        sync.name = name
        return sync

    @staticmethod
    def _make_node(name):
        """Create a mock dbt node with a .name attribute."""
        node = MagicMock()
        node.name = name
        return node

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------

    def test_wires_syncs_after_matching_models(self):
        """Happy path: 2 models with syncs → both wired, correct return dict."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        sync_a = self._make_sync_config("sync_a")
        sync_b = self._make_sync_config("sync_b")
        sync_graph = {"model_a": [sync_a], "model_b": [sync_b]}

        task_a = MagicMock(name="task_a")
        task_b = MagicMock(name="task_b")
        node_a = self._make_node("model_a")
        node_b = self._make_node("model_b")
        nodes_dict = {"unique_a": node_a, "unique_b": node_b}
        tasks_dict = {"unique_a": task_a, "unique_b": task_b}
        tg = self._make_fake_task_group(nodes_dict, tasks_dict)

        with patch(
            "drt.integrations.airflow.DrtRunOperator"
        ) as mock_op_cls, patch(
            "drt.integrations.cosmos.map_syncs_to_manifest",
            return_value=sync_graph,
        ), patch(
            "drt.integrations.cosmos._resolve_manifest",
            return_value={"nodes": {}},
        ), patch(
            "drt.integrations.cosmos._resolve_upstream_task",
            side_effect=lambda v, at: v,
        ):
            mock_op_cls.return_value = MagicMock()
            result = attach_drt_to_cosmos(tg, "/fake/project")

        assert mock_op_cls.call_count == 2
        # Verify >> operator called on each upstream task
        task_a.__rshift__.assert_called_once()
        task_b.__rshift__.assert_called_once()
        assert result == {
            "model_a": ["drt_model_a_sync_a"],
            "model_b": ["drt_model_b_sync_b"],
        }

    def test_task_ids_include_model_name(self):
        """Verify task IDs follow the drt_{model_name}_{sync.name} format."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        sync = self._make_sync_config("my_sync")
        sync_graph = {"orders": [sync]}

        node = self._make_node("orders")
        tg = self._make_fake_task_group(
            {"unique_1": node}, {"unique_1": MagicMock()}
        )

        with patch(
            "drt.integrations.airflow.DrtRunOperator"
        ) as mock_op_cls, patch(
            "drt.integrations.cosmos.map_syncs_to_manifest",
            return_value=sync_graph,
        ), patch(
            "drt.integrations.cosmos._resolve_manifest",
            return_value={"nodes": {}},
        ), patch(
            "drt.integrations.cosmos._resolve_upstream_task",
            side_effect=lambda v, at: v,
        ):
            mock_op_cls.return_value = MagicMock()
            attach_drt_to_cosmos(tg, "/fake/project")

        mock_op_cls.assert_called_once()
        _, kwargs = mock_op_cls.call_args
        assert kwargs["task_id"] == "drt_orders_my_sync"

    def test_empty_sync_graph_returns_empty_dict(self):
        """map_syncs_to_manifest returns {} → function returns {}, no ops."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        tg = self._make_fake_task_group()

        with patch(
            "drt.integrations.airflow.DrtRunOperator"
        ) as mock_op_cls, patch(
            "drt.integrations.cosmos.map_syncs_to_manifest", return_value={}
        ), patch(
            "drt.integrations.cosmos._resolve_manifest",
            return_value={"nodes": {}},
        ):
            result = attach_drt_to_cosmos(tg, "/fake/project")

        assert result == {}
        mock_op_cls.assert_not_called()

    def test_node_not_in_tasks_map_skipped_with_warning(self):
        """Node in filtered_nodes but not tasks_map → logged warning, skipped."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        sync = self._make_sync_config("my_sync")
        sync_graph = {"model_x": [sync]}

        node = self._make_node("model_x")
        # Node NOT in tasks_map
        tg = self._make_fake_task_group({"unique_x": node}, {})

        with patch(
            "drt.integrations.airflow.DrtRunOperator"
        ) as mock_op_cls, patch(
            "drt.integrations.cosmos.map_syncs_to_manifest",
            return_value=sync_graph,
        ), patch(
            "drt.integrations.cosmos._resolve_manifest",
            return_value={"nodes": {}},
        ), patch(
            "drt.integrations.cosmos.logger"
        ) as mock_logger:
            result = attach_drt_to_cosmos(tg, "/fake/project")

        assert result == {}
        mock_op_cls.assert_not_called()
        mock_logger.warning.assert_called_once()
        assert (
            "filtered_nodes but not in tasks_map"
            in mock_logger.warning.call_args[0][0]
        )

    def test_node_without_name_skipped(self):
        """Node with name=None → skipped, no operators created."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        sync = self._make_sync_config("my_sync")
        sync_graph = {"model_x": [sync]}  # won't match name=None

        node = MagicMock()
        node.name = None
        tg = self._make_fake_task_group(
            {"unique_x": node}, {"unique_x": MagicMock()}
        )

        with patch(
            "drt.integrations.airflow.DrtRunOperator"
        ) as mock_op_cls, patch(
            "drt.integrations.cosmos.map_syncs_to_manifest",
            return_value=sync_graph,
        ), patch(
            "drt.integrations.cosmos._resolve_manifest",
            return_value={"nodes": {}},
        ):
            result = attach_drt_to_cosmos(tg, "/fake/project")

        assert result == {}
        mock_op_cls.assert_not_called()

    def test_profile_passed_to_operator(self):
        """profile="my_profile" → DrtRunOperator called with profile kwarg."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        sync = self._make_sync_config("s1")
        sync_graph = {"m": [sync]}
        node = self._make_node("m")
        tg = self._make_fake_task_group(
            {"u1": node}, {"u1": MagicMock()}
        )

        with patch(
            "drt.integrations.airflow.DrtRunOperator"
        ) as mock_op_cls, patch(
            "drt.integrations.cosmos.map_syncs_to_manifest",
            return_value=sync_graph,
        ), patch(
            "drt.integrations.cosmos._resolve_manifest",
            return_value={"nodes": {}},
        ), patch(
            "drt.integrations.cosmos._resolve_upstream_task",
            side_effect=lambda v, at: v,
        ):
            mock_op_cls.return_value = MagicMock()
            attach_drt_to_cosmos(tg, "/p", profile="my_profile")

        _, kwargs = mock_op_cls.call_args
        assert kwargs["profile"] == "my_profile"

    def test_dry_run_passed_to_operator(self):
        """dry_run=True → DrtRunOperator called with dry_run=True."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        sync = self._make_sync_config("s1")
        sync_graph = {"m": [sync]}
        node = self._make_node("m")
        tg = self._make_fake_task_group(
            {"u1": node}, {"u1": MagicMock()}
        )

        with patch(
            "drt.integrations.airflow.DrtRunOperator"
        ) as mock_op_cls, patch(
            "drt.integrations.cosmos.map_syncs_to_manifest",
            return_value=sync_graph,
        ), patch(
            "drt.integrations.cosmos._resolve_manifest",
            return_value={"nodes": {}},
        ), patch(
            "drt.integrations.cosmos._resolve_upstream_task",
            side_effect=lambda v, at: v,
        ):
            mock_op_cls.return_value = MagicMock()
            attach_drt_to_cosmos(tg, "/p", dry_run=True)

        _, kwargs = mock_op_cls.call_args
        assert kwargs["dry_run"] is True

    def test_no_tasks_map_raises_attribute_error(self):
        """Task group without tasks_map → AttributeError with upgrade message."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        # Plain class so hasattr works correctly — MagicMock always hasattr=True
        class _TgNoTasksMap:
            dbt_graph = MagicMock()

        tg = _TgNoTasksMap()

        with pytest.raises(
            AttributeError, match="requires astronomer-cosmos >= 1.8.0"
        ):
            attach_drt_to_cosmos(tg, "/p")

    def test_no_dbt_graph_raises_attribute_error(self):
        """Task group without dbt_graph → AttributeError with upgrade message."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        class _TgNoDbtGraph:
            tasks_map = {}

        tg = _TgNoDbtGraph()

        with pytest.raises(
            AttributeError, match="requires astronomer-cosmos >= 1.8.0"
        ):
            attach_drt_to_cosmos(tg, "/p")

    def test_manifest_passed_to_resolve_manifest(self):
        """Manifest dict is forwarded to _resolve_manifest."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        manifest_data = {"nodes": {"model.test.bar": {"name": "bar"}}}

        sync = self._make_sync_config("s1")
        sync_graph = {"bar": [sync]}
        node = self._make_node("bar")
        tg = self._make_fake_task_group(
            {"u1": node}, {"u1": MagicMock()}
        )

        with patch(
            "drt.integrations.airflow.DrtRunOperator"
        ) as mock_op_cls, patch(
            "drt.integrations.cosmos.map_syncs_to_manifest",
            return_value=sync_graph,
        ), patch(
            "drt.integrations.cosmos._resolve_manifest",
            return_value=manifest_data,
        ) as mock_resolve, patch(
            "drt.integrations.cosmos._resolve_upstream_task",
            side_effect=lambda v, at: v,
        ):
            mock_op_cls.return_value = MagicMock()
            attach_drt_to_cosmos(tg, "/p", manifest=manifest_data)

        mock_resolve.assert_called_once_with(
            manifest=manifest_data,
            project_config=None,
            drt_project_dir=Path("/p"),
        )

    def test_after_tests_passed_to_resolve_upstream(self):
        """_resolve_upstream_task is called with after_tests parameter value."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        sync = self._make_sync_config("s1")
        sync_graph = {"m": [sync]}
        node = self._make_node("m")
        upstream_task = MagicMock(name="upstream")
        tg = self._make_fake_task_group(
            {"u1": node}, {"u1": upstream_task}
        )

        with patch(
            "drt.integrations.airflow.DrtRunOperator"
        ) as mock_op_cls, patch(
            "drt.integrations.cosmos.map_syncs_to_manifest",
            return_value=sync_graph,
        ), patch(
            "drt.integrations.cosmos._resolve_manifest",
            return_value={"nodes": {}},
        ), patch(
            "drt.integrations.cosmos._resolve_upstream_task"
        ) as mock_resolve_upstream:
            mock_resolve_upstream.return_value = upstream_task
            mock_op_cls.return_value = MagicMock()
            attach_drt_to_cosmos(tg, "/p", after_tests=False)

        mock_resolve_upstream.assert_called_once_with(upstream_task, False)

    def test_project_dir_passed_to_operator(self):
        """DrtRunOperator receives project_dir=str(drt_dir)."""
        from drt.integrations.cosmos import attach_drt_to_cosmos

        sync = self._make_sync_config("s1")
        sync_graph = {"m": [sync]}
        node = self._make_node("m")
        tg = self._make_fake_task_group(
            {"u1": node}, {"u1": MagicMock()}
        )

        with patch(
            "drt.integrations.airflow.DrtRunOperator"
        ) as mock_op_cls, patch(
            "drt.integrations.cosmos.map_syncs_to_manifest",
            return_value=sync_graph,
        ), patch(
            "drt.integrations.cosmos._resolve_manifest",
            return_value={"nodes": {}},
        ), patch(
            "drt.integrations.cosmos._resolve_upstream_task",
            side_effect=lambda v, at: v,
        ):
            mock_op_cls.return_value = MagicMock()
            attach_drt_to_cosmos(tg, "/my/project/dir")

        _, kwargs = mock_op_cls.call_args
        assert kwargs["project_dir"] == "/my/project/dir"
