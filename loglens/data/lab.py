"""Lab campaign outputs -> Parquet: lines, incidents, rule-labelled causal lines.

Rule (Implementation Plan 2.5): a line is *causal* if it comes from ``target_service``, falls in
``[t_start, t_end + 60 s]`` and is WARN or above, or matches one of the fault's keywords.
The simulator also records ground truth (``sim_fault``); ``agreement`` reports how well the rule
reproduces it, standing in for the hand check (no human labeller in this pipeline).

Splits are by campaign: last campaign = test, one before = val, rest = train.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from lab.sim.engine import FAULT_KEYWORDS
from loglens.config import load_config, parse_args

WARN_UP = {"WARN", "ERROR", "FATAL", "CRITICAL"}
TAIL_MS = 60_000


@dataclass
class LabDataConfig:
    lab_out: str = "lab/out"
    out_dir: str = "data/lab"
    parquet_dir: str = "data/parquet"
    val_campaigns: int = 1
    test_campaigns: int = 2
    keyword_only_when_info: bool = True


def is_causal_rule(row: dict, inc: dict) -> bool:
    if row["service"] != inc["target_service"]:
        return False
    if not (inc["t_start"] <= row["ts"] <= inc["t_end"] + TAIL_MS):
        return False
    msg = row["message"].lower()
    if row["level"] in WARN_UP:
        return True
    return any(k in msg for k in FAULT_KEYWORDS[inc["fault_type"]])


def load_campaign(run_dir: Path) -> tuple[pl.DataFrame, list[dict]]:
    lines = pl.read_ndjson(run_dir / "logs.jsonl")
    incs = [json.loads(x) for x in (run_dir / "incidents.jsonl").read_text().splitlines() if x]
    return lines.with_columns(pl.lit(run_dir.name).alias("run_id")), incs


def label_campaign(lines: pl.DataFrame, incs: list[dict]) -> tuple[pl.DataFrame, list[dict], list[dict]]:
    n = len(lines)
    inc_id = [None] * n
    causal = [False] * n
    label = [0] * n
    ts = lines["ts"].to_list()
    svc = lines["service"].to_list()
    lvl = lines["level"].to_list()
    msg = lines["message"].to_list()
    truth = lines["sim_fault"].to_list()
    agree = []
    for inc in incs:
        tp = fp = fn = 0
        for i in range(n):
            t = ts[i]
            if t < inc["t_start"] - 60_000:
                continue
            if t > inc["t_end"] + TAIL_MS:
                break
            if inc["t_start"] <= t <= inc["t_end"] + TAIL_MS:
                label[i] = 1
                inc_id[i] = inc["incident_id"]
            row = {"service": svc[i], "level": lvl[i], "message": msg[i], "ts": t}
            c = is_causal_rule(row, inc)
            in_win = inc["t_start"] <= t <= inc["t_end"] + TAIL_MS
            gt = bool(truth[i]) and svc[i] == inc["target_service"] and in_win
            if c and in_win:
                causal[i] = True
            tp += c and gt
            fp += c and not gt
            fn += (not c) and gt
        agree.append({"incident_id": inc["incident_id"], "fault_type": inc["fault_type"],
                      "tp": tp, "fp": fp, "fn": fn})
    df = lines.with_columns(
        pl.Series("incident_id", inc_id, dtype=pl.Utf8),
        pl.Series("is_causal", causal, dtype=pl.Boolean),
        pl.Series("label", label, dtype=pl.Int8),
    )
    return df, incs, agree


def main(cfg: LabDataConfig) -> dict:
    runs = sorted(p for p in Path(cfg.lab_out).iterdir() if (p / "logs.jsonl").exists())
    all_lines, all_incs, all_agree = [], [], []
    for k, r in enumerate(runs):
        lines, incs = load_campaign(r)
        df, incs, agree = label_campaign(lines, incs)
        n_test, n_val = cfg.test_campaigns, cfg.val_campaigns
        split = "test" if k >= len(runs) - n_test else "val" if k >= len(runs) - n_test - n_val else "train"
        all_lines.append(df.with_columns(pl.lit(split).alias("split")))
        all_incs += [{**i, "split": split} for i in incs]
        all_agree += agree
    lines = pl.concat(all_lines).sort("ts")
    lines = lines.with_columns(
        pl.lit("Lab").alias("system"), pl.col("service").alias("component"),
        pl.col("service").alias("host"), pl.col("rid").alias("session_id"),
        pl.int_range(pl.len()).alias("line_no"),
    )
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines.write_parquet(out / "lines.parquet", compression="zstd")
    if cfg.parquet_dir:  # also expose the lab as a regular system for masking / pretraining
        pq = Path(cfg.parquet_dir) / "Lab"
        pq.mkdir(parents=True, exist_ok=True)
        lines.select(
            "system", "ts", "host", "component", "level", "message", "session_id",
            pl.col("label").cast(pl.Int8), pl.col("line_no").cast(pl.Int64), "split",
        ).write_parquet(pq / "part-0000.parquet", compression="zstd")
    pl.DataFrame(all_incs).write_parquet(out / "incidents.parquet")
    causal = lines.filter(pl.col("is_causal")).select(
        "incident_id", "run_id", "ts", "service", "level", "message", "line_no", "split")
    causal.write_parquet(out / "causal_lines.parquet")
    tp = sum(a["tp"] for a in all_agree)
    fp = sum(a["fp"] for a in all_agree)
    fn = sum(a["fn"] for a in all_agree)
    summary = {
        "incidents": len(all_incs),
        "fault_types": sorted({i["fault_type"] for i in all_incs}),
        "lines": len(lines), "causal_lines": len(causal),
        "rule_vs_truth": {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                          "f1": 2 * tp / max(2 * tp + fp + fn, 1)},
        "by_split": {s: sum(1 for i in all_incs if i["split"] == s) for s in ("train", "val", "test")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    args = parse_args("lab parquet + causal labels")
    main(load_config(LabDataConfig, args.config))
