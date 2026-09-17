"""
Tests for models.common.registry: relocatable registry directories.

A registry directory (registry.json + its *.pkl / *_meta.json artifacts) is
seeded once and then mounted/copied to different absolute paths across
environments -- e.g. a macOS host path at seed time vs. a different Linux
bind-mount path inside a container at serve time. Regression coverage for
the bug where save_model wrote absolute model_path/meta_path entries into
registry.json, so a registry that worked where it was seeded raised
FileNotFoundError anywhere else.

Run with: pytest models/tests/test_registry.py -v
"""

from __future__ import annotations

import json
import shutil

from models.common.registry import ModelRegistry

METADATA = {
    "model_type": "diagnostic_test",
    "dataset": "unit_test",
    "metrics": {"test_mae_seconds": 1.0},
}


class TestSaveStoresRelativePaths:
    """save_model must write registry.json entries relative to base_dir."""

    def test_registry_json_entries_are_bare_filenames(self, tmp_path):
        registry = ModelRegistry(base_dir=str(tmp_path))
        registry.save_model("m1", {"value": 1}, dict(METADATA))

        with open(tmp_path / "registry.json") as f:
            on_disk = json.load(f)

        entry = on_disk["m1"]
        assert entry["model_path"] == "m1.pkl"
        assert entry["meta_path"] == "m1_meta.json"

    def test_meta_json_model_path_field_is_also_relative(self, tmp_path):
        # metadata['model_path'] is written into <key>_meta.json too; it
        # should carry the same relocatable convention, not a host absolute
        # path baked into the artifact itself.
        registry = ModelRegistry(base_dir=str(tmp_path))
        registry.save_model("m1", {"value": 1}, dict(METADATA))

        with open(tmp_path / "m1_meta.json") as f:
            meta_on_disk = json.load(f)

        assert meta_on_disk["model_path"] == "m1.pkl"


class TestLoadAfterRelocation:
    """A registry directory must still load after being moved/copied."""

    def test_load_model_after_directory_rename(self, tmp_path):
        original_dir = tmp_path / "reg_original"
        registry = ModelRegistry(base_dir=str(original_dir))
        registry.save_model("m1", {"value": 42}, dict(METADATA))

        relocated_dir = tmp_path / "reg_relocated"
        original_dir.rename(relocated_dir)

        relocated_registry = ModelRegistry(base_dir=str(relocated_dir))
        loaded = relocated_registry.load_model("m1")
        assert loaded == {"value": 42}

    def test_load_metadata_after_directory_rename(self, tmp_path):
        original_dir = tmp_path / "reg_original"
        registry = ModelRegistry(base_dir=str(original_dir))
        registry.save_model("m1", dict(value=1), dict(METADATA))

        relocated_dir = tmp_path / "reg_relocated"
        original_dir.rename(relocated_dir)

        relocated_registry = ModelRegistry(base_dir=str(relocated_dir))
        meta = relocated_registry.load_metadata("m1")
        assert meta["model_type"] == "diagnostic_test"

    def test_load_model_after_directory_copy(self, tmp_path):
        # Simulates the real-world case: the registry is seeded on the host
        # (base_dir = /Users/.../databus/backend/eta_models) and the SAME
        # directory contents are bind-mounted at a different absolute path
        # inside a container (e.g. /app/eta_models). copytree stands in for
        # "mounted somewhere else" since a monkeypatched MODEL_REGISTRY_DIR
        # pointing at a copy is behaviorally identical to a different mount
        # path for a directory whose absolute path changed.
        source_dir = tmp_path / "reg_a"
        registry = ModelRegistry(base_dir=str(source_dir))
        registry.save_model("polyreg_v0", {"coef": [1, 2, 3]}, dict(METADATA))

        mounted_elsewhere = tmp_path / "reg_b"
        shutil.copytree(source_dir, mounted_elsewhere)

        mounted_registry = ModelRegistry(base_dir=str(mounted_elsewhere))
        loaded = mounted_registry.load_model("polyreg_v0")
        assert loaded == {"coef": [1, 2, 3]}

    def test_delete_model_after_directory_rename(self, tmp_path):
        original_dir = tmp_path / "reg_original"
        registry = ModelRegistry(base_dir=str(original_dir))
        registry.save_model("m1", {"value": 1}, dict(METADATA))

        relocated_dir = tmp_path / "reg_relocated"
        original_dir.rename(relocated_dir)

        relocated_registry = ModelRegistry(base_dir=str(relocated_dir))
        assert relocated_registry.delete_model("m1") is True
        assert not (relocated_dir / "m1.pkl").exists()
        assert not (relocated_dir / "m1_meta.json").exists()


class TestLegacyAbsolutePathFallback:
    """
    Registries written before this fix have absolute model_path/meta_path
    entries. load_model/load_metadata must tolerate them: try the absolute
    path as-is first, then fall back to resolving the entry's basename
    against base_dir so the registry loads once the artifact files sit next
    to registry.json (i.e. once the whole directory -- registry.json plus
    the still-legacy-pathed artifacts -- has been relocated as a unit).
    """

    def _write_legacy_registry(self, reg_dir, stale_absolute_dir):
        reg_dir.mkdir(parents=True, exist_ok=True)

        with open(reg_dir / "m1.pkl", "wb") as f:
            import pickle
            pickle.dump({"value": 99}, f)
        with open(reg_dir / "m1_meta.json", "w") as f:
            json.dump({**METADATA, "model_key": "m1"}, f)

        # Entries point at a path that no longer exists (simulates a
        # registry seeded elsewhere and relocated without this fix).
        legacy_registry = {
            "m1": {
                "model_path": str(stale_absolute_dir / "m1.pkl"),
                "meta_path": str(stale_absolute_dir / "m1_meta.json"),
                "saved_at": "2026-01-01T00:00:00",
                "model_type": "diagnostic_test",
                "route_id": None,
                "dataset": "unit_test",
            }
        }
        with open(reg_dir / "registry.json", "w") as f:
            json.dump(legacy_registry, f)

    def test_load_model_falls_back_to_basename_when_absolute_path_missing(self, tmp_path):
        reg_dir = tmp_path / "reg"
        stale_absolute_dir = tmp_path / "does" / "not" / "exist"
        self._write_legacy_registry(reg_dir, stale_absolute_dir)

        registry = ModelRegistry(base_dir=str(reg_dir))
        loaded = registry.load_model("m1")
        assert loaded == {"value": 99}

    def test_load_metadata_falls_back_to_basename_when_absolute_path_missing(self, tmp_path):
        reg_dir = tmp_path / "reg"
        stale_absolute_dir = tmp_path / "does" / "not" / "exist"
        self._write_legacy_registry(reg_dir, stale_absolute_dir)

        registry = ModelRegistry(base_dir=str(reg_dir))
        meta = registry.load_metadata("m1")
        assert meta["model_type"] == "diagnostic_test"

    def test_legacy_absolute_path_used_as_is_when_it_still_exists(self, tmp_path):
        # If the registry hasn't moved, the (still-valid) absolute legacy
        # path should be used directly rather than forcing a relocation.
        real_dir = tmp_path / "reg"
        real_dir.mkdir()

        import pickle
        with open(real_dir / "m1.pkl", "wb") as f:
            pickle.dump({"value": 7}, f)
        with open(real_dir / "m1_meta.json", "w") as f:
            json.dump({**METADATA, "model_key": "m1"}, f)

        legacy_registry = {
            "m1": {
                "model_path": str(real_dir / "m1.pkl"),
                "meta_path": str(real_dir / "m1_meta.json"),
                "saved_at": "2026-01-01T00:00:00",
                "model_type": "diagnostic_test",
                "route_id": None,
                "dataset": "unit_test",
            }
        }
        with open(real_dir / "registry.json", "w") as f:
            json.dump(legacy_registry, f)

        registry = ModelRegistry(base_dir=str(real_dir))
        loaded = registry.load_model("m1")
        assert loaded == {"value": 7}
