"""Per-system data stats + leakage report -> docs/data_report.md."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from loglens.config import load_config, parse_args


@dataclass
class ReportConfig:
    masked_dir: str = "data/masked"
    manifest: str = "data/splits/manifest.json"
    out_md: str = "docs/data_report.md"
    parquet_dir: str = "data/parquet"
    drain: bool = True
    drain_max_unique: int = 120_000


def leakage(system: str, masked_dir: str) -> dict:
    lines = pl.read_parquet(Path(masked_dir) / system / "lines.parquet", columns=["uid", "split"])
    train_uids = set(lines.filter(pl.col("split") == "train")["uid"].unique().to_list())
    test = lines.filter(pl.col("split") == "test")
    if len(test) == 0:
        return {"test_lines_seen_in_train": None, "test_distinct_seen_in_train": None}
    seen = test["uid"].is_in(list(train_uids))
    tu = test["uid"].unique()
    return {"test_lines_seen_in_train": float(seen.mean()),
            "test_distinct_seen_in_train": float(tu.is_in(list(train_uids)).mean())}


def main(cfg: ReportConfig) -> None:
    man = json.loads(Path(cfg.manifest).read_text())
    rows = []
    for system, st in sorted(man["systems"].items()):
        u = pl.read_parquet(Path(cfg.masked_dir) / system / "uniques.parquet")
        avg_len = float(pl.read_parquet(Path(cfg.parquet_dir) / system / "part-0000.parquet",
                                        columns=["message"])["message"].str.len_chars().mean())
        span_h = (st["ts_max"] - st["ts_min"]) / 3.6e6
        d0 = datetime.fromtimestamp(st["ts_min"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        d1 = datetime.fromtimestamp(st["ts_max"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        drain_n = ""
        if cfg.drain:
            from loglens.eval.baselines.classic import drain_clusters

            try:
                drain_n = str(len(set(drain_clusters(system, cfg.masked_dir, cfg.drain_max_unique).tolist())))
            except Exception as e:  # noqa: BLE001
                drain_n = f"err:{type(e).__name__}"
        lk = leakage(system, cfg.masked_dir)
        rows.append({
            "system": system, "lines": st["rows"], "span": f"{d0} → {d1} ({span_h:.0f} h)",
            "anomaly": "-" if st["anomaly_rate"] is None else f"{st['anomaly_rate'] * 100:.2f}%",
            "masked_uniques": len(u), "drain": drain_n, "avg_len": avg_len,
            "dup": st["duplicate_rate"], "trunc": st["truncated"], "dropped": st["dropped_empty"],
            "lk_lines": lk["test_lines_seen_in_train"], "lk_dist": lk["test_distinct_seen_in_train"],
            "unseen": system == man["unseen"],
        })
        print(system, "done", flush=True)
    md = ["# Data report", "",
          f"Split manifest hash `{man.get('hash', '?')}`. Time-based 70/10/20 splits; sessions are "
          "assigned by first timestamp. Unseen (held-out) system: "
          f"**{man['unseen']}** (excluded from tokenizer/encoder/window pretraining).", "",
          "| system | lines | time span | anomaly rate | distinct masked lines | Drain3 clusters | avg chars | "
          "dup rate (raw) | truncated | dropped |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(f"| {r['system']}{' (unseen)' if r['unseen'] else ''} | {r['lines']:,} | {r['span']} | "
                  f"{r['anomaly']} | {r['masked_uniques']:,} | {r['drain']} | {r['avg_len']:.0f} | "
                  f"{r['dup'] * 100:.1f}% | {r['trunc']:,} | {r['dropped']:,} |")
    md += ["", "## Leakage check", "",
           "Fraction of **test** lines whose exact masked text (level + service + masked message) "
           "also appears in the train split. This is reported, not hidden: high values mean a model "
           "can score well by template lookup, so anomaly F1 on those systems overstates "
           "generalisation.", "",
           "| system | test lines seen in train | distinct test masked lines seen in train |",
           "| --- | --- | --- |"]
    for r in rows:
        if r["lk_lines"] is None:
            continue
        md.append(f"| {r['system']} | {r['lk_lines'] * 100:.1f}% | {r['lk_dist'] * 100:.1f}% |")
    Path(cfg.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = parse_args("data report")
    main(load_config(ReportConfig, args.config))
