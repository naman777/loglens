"""Mask every line once and factor each system into (unique masked lines, per-line ids).

Logs are >90% duplicates after masking, so the encoder only ever needs to embed the unique table;
``lines.parquet`` keeps the per-line stream (ts, split, label, session, uid) for window models.

Outputs per system under ``data/masked/<system>/``:
  uniques.parquet: uid, svc, level, masked, count, count_train
  lines.parquet:   line_no, ts, split, label, session_id, host, component, level, uid
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from loglens.config import load_config, parse_args
from loglens.tokenizer.masking import mask


@dataclass
class MaskedConfig:
    parquet_dir: str = "data/parquet"
    out_dir: str = "data/masked"
    systems: list[str] = field(default_factory=list)  # empty = every system found


def build_system(system: str, parquet_dir: str, out_dir: str) -> dict:
    df = pl.read_parquet(Path(parquet_dir) / system / "part-*.parquet")
    raw = df.select("message").unique()
    masked = pl.DataFrame({"message": raw["message"], "masked": [mask(m) for m in raw["message"]]})
    df = df.join(masked, on="message", how="left")
    svc = pl.col("component") if system == "Lab" else pl.lit(system)
    df = df.with_columns(pl.col("level").fill_null("UNK").str.to_uppercase(), svc.alias("svc"))
    key = df.select("svc", "level", "masked").unique().sort(["svc", "level", "masked"])
    key = key.with_row_index("uid").with_columns(pl.col("uid").cast(pl.Int64))
    df = df.join(key, on=["svc", "level", "masked"], how="left")
    counts = df.group_by("uid").agg(
        pl.len().alias("count"),
        (pl.col("split") == "train").sum().alias("count_train"),
    )
    uniques = key.join(counts, on="uid").sort("uid")
    out = Path(out_dir) / system
    out.mkdir(parents=True, exist_ok=True)
    uniques.write_parquet(out / "uniques.parquet", compression="zstd")
    df.select("line_no", "ts", "split", "label", "session_id", "host", "component", "level",
              "uid").write_parquet(out / "lines.parquet", compression="zstd")
    return {"lines": len(df), "unique_masked": len(uniques),
            "unique_frac": round(len(uniques) / len(df), 5)}


def main(cfg: MaskedConfig) -> dict:
    root = Path(cfg.parquet_dir)
    systems = cfg.systems or sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    res = {}
    for s in systems:
        res[s] = build_system(s, cfg.parquet_dir, cfg.out_dir)
        print(s, res[s], flush=True)
    return res


if __name__ == "__main__":
    args = parse_args("mask + factor each system")
    main(load_config(MaskedConfig, args.config))
