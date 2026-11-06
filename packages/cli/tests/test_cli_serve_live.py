"""`stoptimes serve` with a live-feature artifact: session wiring and refusals (synthetic GTFS, no network)."""

from __future__ import annotations

import json
import math
import pickle
import textwrap
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from stoptimes.cli.main import app
from typer.testing import CliRunner

pytest.importorskip("xgboost")
runner = CliRunner()

LAT0, LON0 = 42.0, -71.0
M_DEG = 6371000.0 * math.pi / 180.0
COS = math.cos(math.radians(LAT0))
N_STOPS = 13


def _east(m: float) -> tuple[float, float]:
    return LAT0, LON0 + m / (COS * M_DEG)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("STOPTIMES_ALLOW_UNPRODUCIBLE_FEATURES", raising=False)


def _static_zip(path: Path) -> Path:
    stops = "stop_id,stop_name,stop_lat,stop_lon\n" + "".join(
        f"s{i},S{i},{_east(i * 500.0)[0]},{_east(i * 500.0)[1]}\n" for i in range(N_STOPS)
    )
    shape = "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n" + "".join(
        f"sh,{_east(i * 250.0)[0]},{_east(i * 250.0)[1]},{i}\n" for i in range(2 * N_STOPS - 1)
    )
    times = "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n" + "".join(
        f"trip1,{8 + i // 60:02d}:{i % 60:02d}:00,{8 + i // 60:02d}:{i % 60:02d}:00,s{i},{i + 1}\n" for i in range(N_STOPS)
    )
    files = {
        "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\na,A,http://x,America/New_York\n",
        "routes.txt": "route_id,route_short_name,route_type\nR1,1,3\n",
        "trips.txt": "route_id,service_id,trip_id,shape_id\nR1,wk,trip1,sh\n",
        "stops.txt": stops,
        "stop_times.txt": times,
        "shapes.txt": shape,
    }
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return path


def _live_registry(tmp_path: Path) -> str:
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from stoptimes.core.contracts.artifact import ArtifactMetadata, Provenance, write_asset, write_metadata
    from stoptimes.core.features.identity import RouteVocabulary
    from stoptimes.core.features.live import LIVE_FEATURES
    from stoptimes.core.features.prior import fit

    directory = tmp_path / "registry" / "live-cli"
    stats = fit(pd.DataFrame(columns=["route_id", "from_stop_id", "to_stop_id", "travel_s", "seglen_m", "hour", "daytype", "service_date"]))
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(200, len(LIVE_FEATURES))), columns=list(LIVE_FEATURES))
    model = xgb.XGBRegressor(n_estimators=4, max_depth=2, n_jobs=1).fit(x, rng.normal(300, 50, 200))
    model.required_features = list(LIVE_FEATURES)
    meta = ArtifactMetadata(
        artifact_id="live-cli",
        family="xgboost",
        features=list(LIVE_FEATURES),
        provenance=Provenance(created_at=datetime.now(UTC)),
        timezone="America/New_York",
        holiday_region="US_MA",
        assets={
            "segment_stats": write_asset(directory, "segment_stats.bin", stats.to_bytes()),
            "route_vocabulary": write_asset(
                directory, "route_vocabulary.json", json.dumps(RouteVocabulary.fit(["R1"]).to_dict()).encode()
            ),
        },
    )
    write_metadata(meta, directory)
    with open(directory / meta.model_filename, "wb") as fh:
        pickle.dump(model, fh)
    return "live-cli"


def _sources(tmp_path: Path, *, static_url: str = "unused.zip") -> Path:
    sources = tmp_path / "agencies"
    sources.mkdir(exist_ok=True)
    (sources / "demo.yaml").write_text(
        f"name: demo\ngtfs_static_url: '{static_url}'\npoll_interval_s: 15\ntimezone: UTC\n"
        "vehicle_positions_url: http://example.invalid/vp.json\n"
        "trip_updates_url: http://example.invalid/tu.json\n",
        encoding="utf-8",
    )
    return sources


_PIPELINE = textwrap.dedent(
    """
    import pandas as pd
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from stoptimes.core.contracts.data import Observation, ObservationKind
    from stoptimes.serve.pipeline import build_default_pipeline

    pipeline = build_default_pipeline()

    @pipeline.stage("ingest")
    def _ingest(_payload, ctx):
        # three pings along the line ending 2 s before the tick, on today's service date
        import math
        today = ctx.now.astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d")
        out = []
        for k, back in enumerate((12.0, 7.0, 2.0)):
            m = 600.0 + 120.0 * k
            out.append(Observation(
                kind=ObservationKind.VEHICLE_POSITION, source_id=ctx.source.id, trip_id="trip1",
                route_id="R1", vehicle_id="v1", start_date=today,
                ts=pd.Timestamp(ctx.now.timestamp() - back, unit="s", tz="UTC"),
                lat=42.0, lon=-71.0 + m / (math.cos(math.radians(42.0)) * 6371000.0 * math.pi / 180.0)))
        return out
    """
)


def test_live_artifact_serves_once_through_the_run_session(tmp_path):
    zip_path = _static_zip(tmp_path / "gtfs.zip")
    art = _live_registry(tmp_path)
    pipeline_file = tmp_path / "pipe.py"
    pipeline_file.write_text(_PIPELINE)
    out = tmp_path / "tu.json"
    result = runner.invoke(
        app,
        [
            "serve", "--source", "demo", "--sources-dir", str(_sources(tmp_path)), "--registry",
            str(tmp_path / "registry"), "--artifact", art, "--static", str(zip_path),
            "--pipeline", f"{pipeline_file}:pipeline", "--sink", f"file:{out}", "--once", "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output.strip().splitlines()[-1])
    assert summary["live"] is True and summary["errors"] == []
    assert summary["trip_update_count"] == 1 and summary["published"] is True
    stops = json.loads(out.read_bytes())["entity"][0]["trip_update"]["stop_time_update"]
    assert len(stops) >= 5 and stops[0]["stop_id"] == "s2"  # next stop ahead of 840 m is s2 (1000 m)


def test_live_artifact_without_static_feed_is_refused(tmp_path):
    art = _live_registry(tmp_path)
    result = runner.invoke(
        app,
        ["serve", "--source", "demo", "--sources-dir", str(_sources(tmp_path, static_url="")), "--registry",
         str(tmp_path / "registry"), "--artifact", art, "--once"],
    )
    assert result.exit_code == 1
    assert "GTFS static" in result.output
