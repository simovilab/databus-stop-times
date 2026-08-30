"""
Tests for training provenance (roadmap 3.3): every model saved via
ModelRegistry.save_model must record enough to reproduce it -- git SHA,
library/python versions, a dataset content hash, the random seed, and the
exact feature list -- plus fixed-seed reproducibility for xgboost.

HARD CONSTRAINT: every model trained/saved here goes into a temporary
registry directory (tmp_path), never models/trained/. Nothing here touches
the real registry.json or the served bUCR *.pkl artifacts.

Run with:
    PYTHONPATH=. uv run --group bucr --with pytest python -m pytest \
        models/tests/test_provenance.py -q
"""

from __future__ import annotations

import re

from models.common.data import load_dataset
from models.common.keys import ModelKey
from models.common.provenance import build_provenance, file_content_hash, git_sha, lib_versions
from models.common.registry import ModelRegistry
from models.xgb.train import train_xgboost

DATASET_NAME = "sample"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _small_pre_split():
    """A few hundred rows sliced off the tracked sample dataset, split into
    train/val/test -- big enough for XGBRegressor to fit meaningfully,
    small enough to keep the test fast."""
    dataset = load_dataset(DATASET_NAME)
    dataset.clean_data()
    train_df, val_df, test_df = dataset.temporal_split(train_frac=0.7, val_frac=0.15)
    return train_df.head(300).copy(), val_df.head(80).copy(), test_df.head(80).copy()


def _train_small_xgb(seed_marker: str = "a"):
    """Train an xgboost model (save_model=False) on the small pre-split."""
    pre_split = _small_pre_split()
    return train_xgboost(
        dataset_name=DATASET_NAME,
        save_model=False,
        pre_split=pre_split,
    )


class TestProvenanceHelpers:
    """Unit coverage for the standalone helpers, independent of the registry."""

    def test_git_sha_is_str_or_none(self):
        sha = git_sha()
        assert sha is None or isinstance(sha, str)
        if sha is not None:
            assert len(sha) > 0

    def test_lib_versions_has_expected_keys(self):
        versions = lib_versions()
        for key in ("python", "sklearn", "xgboost", "numpy", "pandas"):
            assert key in versions
        # python is always resolvable (we're running it)
        assert isinstance(versions["python"], str) and versions["python"]

    def test_file_content_hash_missing_file_is_none(self, tmp_path):
        assert file_content_hash(tmp_path / "does_not_exist.parquet") is None

    def test_file_content_hash_is_sha256_hex(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"some bytes")
        digest = file_content_hash(f)
        assert digest is not None
        assert _SHA256_RE.match(digest)

    def test_build_provenance_resolves_tracked_dataset(self):
        prov = build_provenance(dataset_name=DATASET_NAME, seed=42)
        assert prov["dataset_name"] == DATASET_NAME
        assert prov["dataset_sha256"] is not None
        assert _SHA256_RE.match(prov["dataset_sha256"])
        assert prov["seed"] == 42
        assert "created_at" in prov

    def test_build_provenance_unknown_dataset_is_none_not_raise(self):
        prov = build_provenance(dataset_name="definitely_not_a_real_dataset", seed=None)
        assert prov["dataset_sha256"] is None
        assert prov["seed"] is None


class TestProvenanceShapeOnSavedModel:
    """A model saved through the registry gets a well-formed provenance block."""

    def test_saved_meta_json_has_well_formed_provenance(self, tmp_path):
        registry = ModelRegistry(base_dir=str(tmp_path))

        result = _train_small_xgb()
        model, metadata = result["model"], result["metadata"]

        model_key = ModelKey.generate(
            model_type="xgboost",
            dataset_name=DATASET_NAME,
            feature_groups=["temporal", "spatial"],
        )
        registry.save_model(model_key, model, metadata)

        saved_meta = registry.load_metadata(model_key)

        # provenance block present and well-formed
        assert "provenance" in saved_meta
        prov = saved_meta["provenance"]

        assert prov["git_sha"] is None or isinstance(prov["git_sha"], str)

        assert "lib_versions" in prov
        for key in ("python", "sklearn", "xgboost", "numpy", "pandas"):
            assert key in prov["lib_versions"]

        assert isinstance(prov["python_version"], str) and prov["python_version"]

        assert prov["dataset_name"] == DATASET_NAME
        assert prov["dataset_sha256"] is not None
        assert _SHA256_RE.match(prov["dataset_sha256"])

        # xgb's effective random_state (42 by default) is surfaced as the seed
        assert prov["seed"] == metadata["seed"] == 42

        assert "created_at" in prov and prov["created_at"]

        # feature list survives untouched at the top level of metadata
        assert "features" in saved_meta
        assert isinstance(saved_meta["features"], list)
        assert len(saved_meta["features"]) > 0

        # confirm the real registry/served models were never touched
        assert (tmp_path / f"{model_key}_meta.json").exists()

    def test_deterministic_family_records_null_seed(self, tmp_path):
        # historical_mean has no RNG -- its provenance seed should be
        # explicitly null, not a made-up value.
        from models.historical_mean.train import train_historical_mean

        registry = ModelRegistry(base_dir=str(tmp_path))
        train_df, val_df, test_df = _small_pre_split()

        result = train_historical_mean(
            dataset_name=DATASET_NAME,
            save_model=False,
            pre_split=(train_df, val_df, test_df),
        )
        model, metadata = result["model"], result["metadata"]

        model_key = ModelKey.generate(
            model_type="historical_mean",
            dataset_name=DATASET_NAME,
            feature_groups=["temporal", "route"],
        )
        registry.save_model(model_key, model, metadata)

        saved_meta = registry.load_metadata(model_key)
        assert saved_meta["provenance"]["seed"] is None


class TestFixedSeedReproducibility:
    """Training xgboost twice with the same seed on the same split must
    produce identical test metrics (not necessarily byte-identical pickles --
    xgboost pickle bytes can differ across runs even when predictions match)."""

    def test_same_seed_same_split_yields_identical_test_metrics(self):
        pre_split = _small_pre_split()

        result_a = train_xgboost(dataset_name=DATASET_NAME, save_model=False, pre_split=pre_split)
        result_b = train_xgboost(dataset_name=DATASET_NAME, save_model=False, pre_split=pre_split)

        metrics_a = result_a["metadata"]["metrics"]
        metrics_b = result_b["metadata"]["metrics"]

        assert metrics_a.keys() == metrics_b.keys()
        for key in metrics_a:
            assert metrics_a[key] == metrics_b[key], f"metric {key} differs across identical-seed runs"

        # both runs used the same default seed
        assert result_a["metadata"]["seed"] == result_b["metadata"]["seed"] == 42
