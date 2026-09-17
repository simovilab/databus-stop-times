import sys, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str((REPO_ROOT / "models").resolve()))
import pandas as pd
from lstm.train import train_lstm

def day_grouped_split(dataset, service_offset_h, min_segs, test_days, val_days):
    df = pd.read_parquet(REPO_ROOT / "datasets" / f"{dataset}.parquet")
    df["cr_day"] = (df["vp_ts"] - pd.Timedelta(hours=service_offset_h)).dt.date
    counts = df.groupby("cr_day").size()
    usable = sorted(counts[counts >= min_segs].index)
    test = usable[-test_days:]
    val = usable[-(test_days + val_days):-test_days]
    train = usable[:-(test_days + val_days)]
    return (
        df[df["cr_day"].isin(train)].copy(),
        df[df["cr_day"].isin(val)].copy(),
        df[df["cr_day"].isin(test)].copy(),
    )

CONFIGS = {
    "mbta": dict(dataset="mbta_segments_corpus", service_offset_h=7, min_segs=200, test_days=3, val_days=3),
    "bucr": dict(dataset="bucr_segments_corpus", service_offset_h=6, min_segs=50, test_days=5, val_days=5),
}

if __name__ == "__main__":
    agency = sys.argv[1]
    transform = sys.argv[2] if len(sys.argv) > 2 else "none"
    cfg = CONFIGS[agency]
    tr, va, te = day_grouped_split(cfg["dataset"], cfg["service_offset_h"], cfg["min_segs"], cfg["test_days"], cfg["val_days"])
    t0 = time.time()
    out = train_lstm(
        dataset_name=cfg["dataset"],
        pre_split=(tr, va, te),
        service_offset_h=cfg["service_offset_h"],
        hidden_size=64, num_layers=1, dropout=0.1, rnn_type="lstm",
        learning_rate=1e-3, batch_size=64, max_epochs=300, patience=25,
        weight_decay=0.0, loss="huber", target_transform=transform,
        save_model=False, seed=42,
    )
    dt = time.time() - t0
    m = out["metrics"]
    print(f"\n=== {agency} transform={transform} wall={dt:.0f}s epochs={out['metadata']['epochs_trained']} best_epoch={out['metadata']['best_epoch']} ===")
    print(f"val_mae={m.get('val_mae_seconds'):.2f} test_mae={m.get('test_mae_seconds'):.2f} test_r2={m.get('test_r2'):.3f}")
