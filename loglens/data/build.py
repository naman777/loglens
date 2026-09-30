"""Raw Loghub dirs -> cleaned, time-split Parquet shards + split manifest.

Cleaning: empty / non-UTF-8 lines dropped (counted), messages truncated to ``max_chars`` (counted),
duplicates kept (rate recorded). Split: by time, 70/10/20; sessions (HDFS blocks, Hadoop/Spark apps,
OpenStack instances) are assigned by their first timestamp so no session straddles two splits.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from loglens.config import load_config, parse_args
from loglens.data.parsers.systems import PARSERS
from loglens.seed import set_seed

SCHEMA = {
    "system": pl.Utf8, "ts": pl.Int64, "host": pl.Utf8, "component": pl.Utf8, "level": pl.Utf8,
    "message": pl.Utf8, "session_id": pl.Utf8, "label": pl.Int8, "line_no": pl.Int64,
}


@dataclass
class DataConfig:
    seed: int = 1337
    raw_dir: str = "data/raw"
    out_dir: str = "data/parquet"
    split_dir: str = "data/splits"
    systems: list[str] = field(default_factory=lambda: list(PARSERS))
    shard_rows: int = 1_000_000
    max_chars: int = 1024
    max_lines: dict[str, int] = field(default_factory=dict)  # per-system cap (0/absent = none)
    train_frac: float = 0.7
    val_frac: float = 0.1
    unseen: str = "Thunderbird"


def _flush(rows: list[dict], path: Path) -> None:
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(path, compression="zstd")


def parse_system(system: str, cfg: DataConfig, tmp: Path) -> dict:
    """Stream-parse one system into unsorted raw shards; return cleaning counts."""
    raw = Path(cfg.raw_dir) / system
    stats = {"parsed": 0, "dropped_empty": 0, "truncated": 0, "unparsed_header": 0}
    rows: list[dict] = []
    shard = 0
    last_ts: int | None = None
    for r in PARSERS[system](raw):
        cap = cfg.max_lines.get(system, 0)
        if cap and stats["parsed"] >= cap:
            break
        stats["parsed"] += 1
        msg = r["message"]
        if not msg or not msg.strip():
            stats["dropped_empty"] += 1
            continue
        if r["ts"] is None:  # continuation / unparsed header: inherit previous timestamp
            stats["unparsed_header"] += 1
            if last_ts is None:
                stats["dropped_empty"] += 1
                continue
            r["ts"] = last_ts
        last_ts = r["ts"]
        if len(msg) > cfg.max_chars:
            r["message"] = msg[: cfg.max_chars]
            stats["truncated"] += 1
        rows.append(r)
        if len(rows) >= cfg.shard_rows:
            _flush(rows, tmp / f"raw-{shard:04d}.parquet")
            shard += 1
            rows = []
    if rows:
        _flush(rows, tmp / f"raw-{shard:04d}.parquet")
    return stats


def assign_splits(df: pl.DataFrame, cfg: DataConfig) -> pl.DataFrame:
    """Add ``key_ts`` (session start or own ts) and ``split``; thresholds are row-quantiles of key_ts."""
    df = df.with_columns(
        pl.when(pl.col("session_id").is_not_null())
        .then(pl.col("ts").min().over("session_id"))
        .otherwise(pl.col("ts"))
        .alias("key_ts")
    )
    sorted_keys = df["key_ts"].sort()
    n = len(sorted_keys)
    t_train = sorted_keys[int(n * cfg.train_frac)]
    t_val = sorted_keys[int(n * (cfg.train_frac + cfg.val_frac))]
    return df.with_columns(
        pl.when(pl.col("key_ts") < t_train).then(pl.lit("train"))
        .when(pl.col("key_ts") < t_val).then(pl.lit("val"))
        .otherwise(pl.lit("test")).alias("split")
    )


def build_system(system: str, cfg: DataConfig) -> dict:
    out = Path(cfg.out_dir) / system
    tmp = Path(cfg.out_dir) / f"_tmp_{system}"
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    out.mkdir(parents=True)
    stats = parse_system(system, cfg, tmp)
    df = pl.read_parquet(tmp / "raw-*.parquet").sort(["ts", "line_no"], maintain_order=True)
    df = assign_splits(df, cfg)
    stats["rows"] = len(df)
    stats["duplicate_rate"] = round(1 - df["message"].n_unique() / max(len(df), 1), 4)
    stats["anomaly_rate"] = (round(float(df["label"].drop_nulls().mean()), 5)
                             if df["label"].null_count() < len(df) else None)
    stats["ts_min"], stats["ts_max"] = int(df["ts"].min()), int(df["ts"].max())
    stats["splits"] = {}
    for s in ("train", "val", "test"):
        part = df.filter(pl.col("split") == s)
        if len(part):
            stats["splits"][s] = {
                "rows": len(part), "key_ts_min": int(part["key_ts"].min()),
                "key_ts_max": int(part["key_ts"].max()),
                "sessions": int(part["session_id"].n_unique()) if part["session_id"].null_count() < len(part) else 0,
            }
    for i in range(0, len(df), cfg.shard_rows):
        df.slice(i, cfg.shard_rows).write_parquet(out / f"part-{i // cfg.shard_rows:04d}.parquet",
                                                  compression="zstd")
    shutil.rmtree(tmp, ignore_errors=True)
    return stats


def check_no_overlap(manifest: dict) -> None:
    for name, s in manifest["systems"].items():
        sp = s["splits"]
        if "train" in sp and "test" in sp:
            assert sp["train"]["key_ts_max"] < sp["test"]["key_ts_min"], f"{name}: train/test overlap"
        if "train" in sp and "val" in sp:
            assert sp["train"]["key_ts_max"] < sp["val"]["key_ts_min"], f"{name}: train/val overlap"


def main(cfg: DataConfig) -> dict:
    set_seed(cfg.seed)
    Path(cfg.split_dir).mkdir(parents=True, exist_ok=True)
    stats_dir = Path(cfg.split_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for system in cfg.systems:
        if not (Path(cfg.raw_dir) / system).exists():
            print(f"skip {system}: not downloaded")
            continue
        print(f"building {system} ...", flush=True)
        st = build_system(system, cfg)
        (stats_dir / f"{system}.json").write_text(json.dumps(st, indent=2, sort_keys=True))
        print(" ", {k: v for k, v in st.items() if k != "splits"}, flush=True)
    manifest: dict = {"unseen": cfg.unseen, "systems": {}}
    for f in sorted(stats_dir.glob("*.json")):
        manifest["systems"][f.stem] = json.loads(f.read_text())
    check_no_overlap(manifest)
    body = json.dumps(manifest, indent=2, sort_keys=True)
    manifest["hash"] = hashlib.sha256(body.encode()).hexdigest()[:12]
    Path(cfg.split_dir, "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    args = parse_args("build parquet + splits")
    main(load_config(DataConfig, args.config))
