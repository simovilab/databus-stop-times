"""Tests for core.catalog.store.Catalog and DatasetEntry."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from stoptimes.core.catalog import Catalog, DatasetEntry, DatasetExistsError, DatasetKind
from stoptimes.core.catalog.store import DatasetNotFoundError, catalog_path


def _entry(name="mbta_vp", **overrides) -> DatasetEntry:
    fields = dict(
        name=name,
        source_id="mbta",
        kind=DatasetKind.OBSERVATIONS,
        location="/data/mbta/vp",
        added_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    fields.update(overrides)
    return DatasetEntry(**fields)


def test_add_then_get_round_trips(tmp_path):
    catalog = Catalog(tmp_path / "catalog.json")
    catalog.add(_entry())
    fetched = catalog.get("mbta_vp")
    assert fetched is not None
    assert fetched.source_id == "mbta"
    assert fetched.kind is DatasetKind.OBSERVATIONS


def test_get_missing_returns_none(tmp_path):
    catalog = Catalog(tmp_path / "catalog.json")
    assert catalog.get("nope") is None


def test_resolve_missing_raises(tmp_path):
    catalog = Catalog(tmp_path / "catalog.json")
    with pytest.raises(DatasetNotFoundError):
        catalog.resolve("nope")


def test_resolve_returns_location(tmp_path):
    catalog = Catalog(tmp_path / "catalog.json")
    catalog.add(_entry())
    assert catalog.resolve("mbta_vp") == "/data/mbta/vp"


def test_add_duplicate_name_raises_without_overwrite(tmp_path):
    catalog = Catalog(tmp_path / "catalog.json")
    catalog.add(_entry())
    with pytest.raises(DatasetExistsError):
        catalog.add(_entry(location="/other/path"))


def test_add_duplicate_name_with_overwrite_replaces(tmp_path):
    catalog = Catalog(tmp_path / "catalog.json")
    catalog.add(_entry())
    catalog.add(_entry(location="/other/path"), overwrite=True)
    assert catalog.get("mbta_vp").location == "/other/path"


def test_list_is_sorted_by_name(tmp_path):
    catalog = Catalog(tmp_path / "catalog.json")
    catalog.add(_entry(name="zzz"))
    catalog.add(_entry(name="aaa"))
    names = [e.name for e in catalog.list()]
    assert names == ["aaa", "zzz"]


def test_remove_existing_returns_true_and_removes(tmp_path):
    catalog = Catalog(tmp_path / "catalog.json")
    catalog.add(_entry())
    assert catalog.remove("mbta_vp") is True
    assert catalog.get("mbta_vp") is None


def test_remove_missing_returns_false(tmp_path):
    catalog = Catalog(tmp_path / "catalog.json")
    assert catalog.remove("nope") is False


def test_write_is_atomic_no_leftover_temp_files(tmp_path):
    catalog_file = tmp_path / "catalog.json"
    catalog = Catalog(catalog_file)
    catalog.add(_entry())
    catalog.add(_entry(name="second"))
    leftover_temp_files = list(tmp_path.glob(".catalog.*"))
    assert leftover_temp_files == []
    assert catalog_file.exists()


def test_catalog_persists_across_instances(tmp_path):
    path = tmp_path / "catalog.json"
    Catalog(path).add(_entry())
    reopened = Catalog(path)
    assert reopened.get("mbta_vp") is not None


def test_catalog_path_prefers_stoptimes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path))
    from stoptimes.core.config import reset_config

    reset_config()
    try:
        assert catalog_path() == tmp_path / "catalog.json"
    finally:
        reset_config()


def test_dataset_entry_rejects_unknown_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _entry(bogus_field="nope")
