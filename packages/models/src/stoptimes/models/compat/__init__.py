"""Compatibility shims for pre-migration (gtfs-eta) model artifacts.

`stoptimes.models.compat.migrate` turns a legacy flat registry
(`{key}.pkl` + `{key}_meta.json` + `registry.json`) into the fixed
artifact layout from `stoptimes.core.contracts.artifact`.
"""

from .migrate import MigrationItem, MigrationReport, migrate, plan_migration

__all__ = ["MigrationItem", "MigrationReport", "migrate", "plan_migration"]
