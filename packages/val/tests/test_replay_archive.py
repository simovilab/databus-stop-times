

def test_archive_opens_when_a_partition_key_is_also_a_column(tmp_path):
    """pandas >= 3 writes `large_string` columns; hive partition values parse
    as `string`. Opening the dataset must not fail on that mismatch.
    """
    import pandas as pd

    from stoptimes.val.feeds.replay_archive import _read_table

    partition = tmp_path / "year=2026" / "month=09" / "day=17" / "route_id=r1"
    partition.mkdir(parents=True)
    pd.DataFrame(
        {"route_id": ["r1", "r1"], "trip_id": ["t1", "t2"], "ts": [1.0, 2.0]}
    ).to_parquet(partition / "part-0.parquet")

    table = _read_table(tmp_path)
    assert len(table) == 2
    assert set(table["route_id"]) == {"r1"}


def test_archive_with_timestamp_column_is_windowed_correctly(tmp_path):
    """The recorder writes `ts` as a tz-aware timestamp; replay used to coerce
    it with `to_numeric`, silently turning every row into NaT and reporting an
    empty archive."""
    import pandas as pd

    from stoptimes.val.feeds.replay_archive import read_vp_groups

    partition = tmp_path / "year=2026" / "month=09" / "day=17" / "route_id=r1"
    partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "route_id": ["r1", "r1"],
            "trip_id": ["t1", "t1"],
            "vehicle_id": ["v1", "v1"],
            "ts": [
                pd.Timestamp("2026-09-17T08:00:00Z"),
                pd.Timestamp("2026-09-17T08:00:30Z"),
            ],
            "lat": [0.0, 0.0],
            "lon": [0.0, 0.001],
        }
    ).to_parquet(partition / "part-0.parquet")

    rows = read_vp_groups(tmp_path, start="2026-09-17", end="2026-09-18")
    assert len(rows) == 2
    assert rows["ts"].tolist() == [1789632000.0, 1789632030.0]
