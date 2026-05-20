"""Cosmos integration — auto-wire drt syncs after dbt model tasks.

Uses drt's own ``map_syncs_to_manifest()`` to discover which drt syncs
depend on which dbt models, then creates ``DrtRunOperator`` tasks and
wires them after the matching Cosmos dbt model tasks.

Requires:
    - astronomer-cosmos >= 1.8.0 (for ``tasks_map`` property)
    - apache-airflow (for DrtRunOperator)

Usage::

    from cosmos import DbtTaskGroup, ProjectConfig, ProfileConfig
    from drt.integrations.cosmos import attach_drt_to_cosmos

    dbt = DbtTaskGroup(
        group_id="dbt",
        project_config=ProjectConfig(
            dbt_project_path=DBT_PROJECT_PATH,
            manifest_path="/path/to/target/manifest.json",
        ),
        profile_config=profile_config,
    )

    # Pass the same ProjectConfig — drt reads manifest_path from it
    attach_drt_to_cosmos(
        dbt,
        drt_project_dir=DBT_PROJECT_PATH,
        project_config=dbt.project_config,
    )

The ``ref('model_name')`` in each drt sync YAML is the link — no manual
mapping needed. One dbt model can have multiple drt syncs (e.g., to Doris
and HubSpot), each wired independently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from drt.integrations.dbt import map_syncs_to_manifest

__all__ = ["attach_drt_to_cosmos"]

logger = logging.getLogger(__name__)


def attach_drt_to_cosmos(
    dbt_task_group: Any,
    drt_project_dir: str | Path,
    manifest: dict[str, Any] | None = None,
    project_config: Any | None = None,
    profile: str | None = None,
    after_tests: bool = True,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Attach drt sync tasks after corresponding dbt model tasks in Cosmos.

    Iterates over the Cosmos dbt graph, finds models that have drt syncs
    referencing them via ``ref()``, and creates ``DrtRunOperator`` tasks
    wired after each matching model.

    Manifest resolution (pick **one** — no mixing):

    1. ``manifest=dict`` — use the parsed dict directly.
    2. ``project_config=ProjectConfig(...)`` — read from
       ``project_config.manifest_path`` (Cosmos ``ProjectConfig`` instance).
       Cosmos is imported lazily — not a hard dependency.
    3. Neither — read from ``{drt_project_dir}/target/manifest.json``.

    Args:
        dbt_task_group: A Cosmos ``DbtTaskGroup`` or ``DbtDag`` instance.
            Must have ``dbt_graph.filtered_nodes`` and ``tasks_map``
            attributes (Cosmos >= 1.8.0).
        drt_project_dir: Path to the drt project directory. Can be the same
            directory as the dbt project — drt reads ``syncs/*.yml`` and
            ``target/manifest.json``.
        manifest: Parsed ``manifest.json`` dict. Takes priority over
            ``project_config`` and disk read.
        project_config: Cosmos ``ProjectConfig`` instance. drt reads
            ``manifest_path`` from it. Lazy import — Cosmos is **not** a
            hard dependency. Only used when ``manifest`` is *None*.
        profile: Override drt profile name (default: from
            ``drt_project.yml``).
        after_tests: If *True* (default), wire drt tasks after the dbt
            **test** task when one exists (i.e. after both model run AND
            test pass). If *False*, wire immediately after the model run
            task — faster but doesn't wait for test validation.

            .. note::

               This parameter only takes effect with Cosmos
               ``TestBehavior.AFTER_EACH`` (the default). With
               ``TestBehavior.AFTER_ALL``, tests run as separate top-level
               tasks — not nested inside model TaskGroups — so drt cannot
               detect or wait for them. Use Airflow task dependencies
               manually in that case.
        dry_run: If *True*, pass ``dry_run=True`` to each
            ``DrtRunOperator``. Produces a diff preview without writing
            data.

    Returns:
        Dict mapping dbt model name → list of created Airflow task IDs.

    Raises:
        ImportError: If Airflow is not installed.
        FileNotFoundError: If manifest cannot be resolved.
        TypeError: If ``project_config`` is not a Cosmos ``ProjectConfig``.
        ValueError: If ``project_config.manifest_path`` is a remote URI.
    """
    from drt.integrations.airflow import DrtRunOperator

    drt_dir = Path(drt_project_dir)

    # Cosmos version guard — tasks_map and dbt_graph require astronomer-cosmos >= 1.8.0
    if not hasattr(dbt_task_group, "tasks_map") or not hasattr(
        dbt_task_group, "dbt_graph"
    ):
        raise AttributeError(
            "attach_drt_to_cosmos requires astronomer-cosmos >= 1.8.0 "
            "(for the tasks_map and dbt_graph properties). Upgrade with: "
            "pip install astronomer-cosmos>=1.8.0"
        )

    # Resolve manifest — three explicit paths, no cascading
    manifest_data = _resolve_manifest(
        manifest=manifest,
        project_config=project_config,
        drt_project_dir=drt_dir,
    )

    # drt resolves syncs against the manifest using its own APIs
    sync_graph = map_syncs_to_manifest(manifest_data, drt_dir)

    if not sync_graph:
        return {}

    created: dict[str, list[str]] = {}

    for unique_id, dbt_node in dbt_task_group.dbt_graph.filtered_nodes.items():
        model_name = getattr(dbt_node, "name", None)
        if not model_name:
            continue

        syncs_for_model = sync_graph.get(model_name)
        if not syncs_for_model:
            continue

        # Guard against nodes in filtered_nodes but missing from tasks_map
        upstream_value = dbt_task_group.tasks_map.get(unique_id)
        if upstream_value is None:
            logger.warning(
                "Model %s (unique_id=%s) found in filtered_nodes but not "
                "in tasks_map; skipping drt wiring for this model",
                model_name,
                unique_id,
            )
            continue

        upstream_task = _resolve_upstream_task(upstream_value, after_tests)

        for sync in syncs_for_model:
            task_id = f"drt_{model_name}_{sync.name}"
            drt_op = DrtRunOperator(
                task_id=task_id,
                sync_name=sync.name,
                project_dir=str(drt_dir),
                profile=profile,
                dry_run=dry_run,
            )
            upstream_task >> drt_op
            created.setdefault(model_name, []).append(task_id)

    return created


def _resolve_manifest(
    manifest: dict[str, Any] | None,
    project_config: Any | None,
    drt_project_dir: Path,
) -> dict[str, Any]:
    """Resolve manifest dict from one of three sources.

    Priority (pick one — no mixing):

    1. ``manifest`` dict — returned as-is.
    2. ``project_config`` — Cosmos ``ProjectConfig`` with ``manifest_path``.
    3. Disk — ``{drt_project_dir}/target/manifest.json``.
    """
    # 1. Explicit dict wins
    if manifest is not None:
        if project_config is not None:
            logger.warning(
                "Both manifest and project_config provided; using manifest dict."
            )
        return manifest

    # 2. Cosmos ProjectConfig
    if project_config is not None:
        return _read_manifest_from_project_config(project_config)

    # 3. Disk fallback
    manifest_path = drt_project_dir / "target" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found at {manifest_path}. "
            "Pass manifest=dict, project_config=ProjectConfig(...), "
            "or ensure dbt has been compiled."
        )
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"manifest.json at {manifest_path} is not valid JSON: {e}. "
            "This usually means dbt compile was interrupted. "
            "Re-run dbt compile and try again."
        ) from e


def _read_manifest_from_project_config(config: Any) -> dict[str, Any]:
    """Read manifest.json from a Cosmos ``ProjectConfig.manifest_path``.

    Lazy-imports ``cosmos.config.ProjectConfig`` for isinstance validation.
    Cosmos is **not** a hard dependency of drt.
    """
    try:
        from cosmos.config import ProjectConfig  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "project_config requires astronomer-cosmos installed. "
            "Install it with: pip install astronomer-cosmos"
        ) from e

    if not isinstance(config, ProjectConfig):
        raise TypeError(
            f"project_config must be a cosmos.config.ProjectConfig, "
            f"got {type(config).__name__}"
        )

    raw_path = getattr(config, "manifest_path", None)
    if raw_path is None:
        raise FileNotFoundError(
            "ProjectConfig.manifest_path is None. "
            "Set it in your Cosmos ProjectConfig: "
            'ProjectConfig(manifest_path="/path/to/target/manifest.json")'
        )

    # Remote URIs (s3://, gs://, abfss://, etc.) — Cosmos supports these but drt doesn't
    path_str = str(raw_path)
    if "://" in path_str and not path_str.startswith("file://"):
        raise ValueError(
            f"ProjectConfig.manifest_path is a remote URI ({path_str!r}). "
            "drt only supports local manifest paths. "
            "Download the manifest first or pass manifest=dict."
        )

    manifest_path = Path(path_str)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"ProjectConfig.manifest_path points to {manifest_path}, "
            "which does not exist. Run dbt compile or dbt build first."
        )

    logger.info("Reading manifest from ProjectConfig.manifest_path=%s", manifest_path)
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"manifest.json at {manifest_path} is not valid JSON: {e}. "
            "This usually means dbt compile was interrupted. "
            "Re-run dbt compile and try again."
        ) from e


def _resolve_upstream_task(task_or_group: Any, after_tests: bool) -> Any:
    """Find the correct upstream task to wire drt after.

    With Cosmos ``TestBehavior.AFTER_EACH`` (default), models with tests
    are wrapped in a TaskGroup containing ``run`` and ``test`` child tasks.
    Models without tests are plain tasks.

    - ``after_tests=True`` → return the test task if it exists, else the
      run task. This ensures drt runs only after dbt validation passes.
    - ``after_tests=False`` → return the task/group as-is. When it's a
      TaskGroup, Airflow wires after the group completes (same net effect
      as after_tests=True for groups). When it's a plain task, wires
      immediately after the model run.
    """
    if not after_tests:
        return task_or_group

    # If it's a TaskGroup (model has tests), try to get the test task
    try:
        # NOTE: This import path (airflow.utils.task_group.TaskGroup) has
        # been stable since Airflow 2.1 and is used by Cosmos internally.
        # If Airflow reorganizes this module in a future major version,
        # the isinstance check silently fails (safe fallback: returns the
        # group as-is, which still triggers after group completion).
        from airflow.utils.task_group import TaskGroup  # type: ignore[import-untyped]

        if isinstance(task_or_group, TaskGroup):
            # Model has tests — TaskGroup contains {run, test}
            test_task = task_or_group.get_child_by_label("test")
            if test_task is not None:
                return test_task
    except ImportError:
        pass  # Airflow not installed — shouldn't happen at DAG render time

    # No tests, or not a TaskGroup — return as-is
    return task_or_group
