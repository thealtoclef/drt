"""dbt integration — resolve ref() from dbt manifest.json.

When a dbt project is co-located with a drt project, drt can read
target/manifest.json to resolve ref('model_name') to the fully-qualified
table name that dbt materialized.

Usage:
    from drt.integrations.dbt import resolve_ref_from_manifest
    table = resolve_ref_from_manifest("my_model", project_dir)
    # Returns: '"analytics"."public"."my_model"' or None

Cosmos integration:
    from drt.integrations.dbt import map_syncs_to_manifest
    graph = map_syncs_to_manifest(manifest_dict, project_dir)
    # Returns: {"dim_users": [SyncConfig, ...], "fct_orders": [SyncConfig, ...]}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drt.config.models import SyncConfig

logger = logging.getLogger(__name__)


@dataclass
class DbtModel:
    """A model extracted from dbt manifest.json."""

    name: str
    relation_name: str | None
    description: str
    resource_type: str


def list_models_from_manifest(
    manifest_path: Path,
) -> list[DbtModel]:
    """List all models from a dbt manifest.json.

    Returns a list of DbtModel with name, relation_name, and description.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"manifest.json at {manifest_path} is not valid JSON: {e}. "
            "This usually means dbt compile was interrupted. "
            "Re-run dbt compile and try again."
        ) from e
    nodes = manifest.get("nodes", {})

    models: list[DbtModel] = []
    for node in nodes.values():
        if node.get("resource_type") != "model":
            continue
        models.append(
            DbtModel(
                name=node.get("name", ""),
                relation_name=node.get("relation_name"),
                description=node.get("description", ""),
                resource_type=node.get("resource_type", "model"),
            )
        )

    return sorted(models, key=lambda m: m.name)


def resolve_ref_from_manifest(
    model_name: str,
    project_dir: Path,
    manifest_path: Path | None = None,
) -> str | None:
    """Resolve a model name to a fully-qualified table using dbt manifest.

    Looks for target/manifest.json in the project directory.
    Returns the relation_name if found, None otherwise.
    """
    mpath = manifest_path or (project_dir / "target" / "manifest.json")
    if not mpath.exists():
        return None

    try:
        manifest = json.loads(mpath.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"manifest.json at {mpath} is not valid JSON: {e}. "
            "This usually means dbt compile was interrupted. "
            "Re-run dbt compile and try again."
        ) from e
    nodes = manifest.get("nodes", {})

    for node in nodes.values():
        if node.get("name") == model_name:
            rel: str | None = node.get("relation_name")
            return rel

    return None


def map_syncs_to_manifest(
    manifest: dict[str, Any],
    project_dir: Path,
) -> dict[str, list[SyncConfig]]:
    """Map dbt model names to drt syncs that reference them.

    Uses the manifest's node structure to resolve which drt syncs depend on
    which dbt models. The manifest dict is typically provided by Cosmos
    (which already has it parsed in memory).

    Args:
        manifest: Parsed manifest.json dict (from Cosmos or read from disk).
            Must contain a ``nodes`` key with dbt node entries.
        project_dir: drt project directory containing ``syncs/*.yml``.

    Returns:
        Dict mapping dbt model names to lists of SyncConfig instances.

    Example::

        manifest = json.loads("target/manifest.json")
        graph = map_syncs_to_manifest(manifest, Path("."))
        # {"dim_users": [SyncConfig(name="users_to_doris"), ...]}
    """
    from drt.config.parser import load_syncs_safe
    from drt.engine.resolver import parse_ref

    # Build lookup from manifest: model name → True (confirmed exists)
    manifest_models: set[str] = set()
    for node in manifest.get("nodes", {}).values():
        if node.get("resource_type") == "model":
            name = node.get("name")
            if name:
                manifest_models.add(name)

    # Load drt syncs via safe parser — bad YAML files produce warnings,
    # not crashes. This is critical for Airflow DAG parsing where a single
    # ValidationError would make the entire DAG disappear from the UI.
    result = load_syncs_safe(project_dir)
    syncs = result.syncs

    # Log warnings for malformed sync files (don't crash the orchestrator)
    for filename, errors in result.errors.items():
        for err in errors:
            logger.warning("Skipping drt sync %s: %s", filename, err)

    # Resolve each sync's model field against the manifest
    mapping: dict[str, list[SyncConfig]] = {}
    for sync in syncs:
        model_name = parse_ref(sync.model)
        if model_name is None:
            continue  # raw SQL, not a ref()

        # Strip to handle accidental whitespace inside ref() quotes
        model_name = model_name.strip()

        if model_name in manifest_models:
            mapping.setdefault(model_name, []).append(sync)

    return mapping
