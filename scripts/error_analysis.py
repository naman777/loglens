"""Dump the worst LogLens misses per experiment for manual labelling -> results/error_analysis.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loglens.eval.report import load_window_model  # noqa: E402
from loglens.seed import get_device  # noqa: E402
from loglens.train.finetune import rank_incident, sup_scores  # noqa: E402
from loglens.train.windata import SystemData  # noqa: E402


def uniq(system):
    return pl.read_parquet(f"data/masked/{system}/uniques.parquet").sort("uid")


def main() -> None:
    dev = get_device()
    m, fc = load_window_model("artifacts/window/sup/best.pt", dev)
    thr = json.loads(Path("results/results.json").read_text())["anomaly"]
    out = {}
    # 1. BGL window misses
    sd = SystemData("BGL", "data/win", "data/emb")
    u = uniq("BGL")
    wt = sd.windows("test")
    s = sup_scores(m, sd, wt, fc, dev)
    t = thr["BGL"]["LogLens-sup"]["threshold"]
    y = sd.z["wlabel"][wt]
    fn = np.nonzero((y == 1) & (s < t))[0]
    fp = np.nonzero((y == 0) & (s >= t))[0]
    def window_dump(wi, sd, u):
        a, b = sd.span(int(wi))
        lab = sd.z["label"][a:b]
        return {"window": int(wi), "alert_lines": [u["masked"][int(x)][:120] for x in sd.z["uid"][a:b][lab == 1][:3]],
                "sample_lines": [u["masked"][int(x)][:120] for x in sd.z["uid"][a:b][:3]]}
    worst_fn = fn[np.argsort(s[fn])[:30]]
    worst_fp = fp[np.argsort(-s[fp])[:30]]
    out["BGL_false_negatives"] = [window_dump(wt[i], sd, u) for i in worst_fn]
    out["BGL_false_positives"] = [window_dump(wt[i], sd, u) for i in worst_fp]
    out["BGL_counts"] = {"fn": int(len(fn)), "fp": int(len(fp)), "threshold": t}
    # 2. lab RCA misses
    sd = SystemData("Lab", "data/win", "data/emb")
    u = uniq("Lab")
    for k, (iid, ftype, tgt, sp) in enumerate(sd.incidents):
        if sp != "test":
            continue
        a, b = int(sd.z["inc_a"][k]), int(sd.z["inc_b"][k])
        sc = rank_incident(m, sd, a, b, fc, dev)
        causal = sd.z["causal"][a:b].astype(bool)
        order = np.argsort(-sc)
        best_rank = int(np.nonzero(causal[order])[0][0]) + 1 if causal.any() else None
        if best_rank is None or best_rank > 5:
            uid = sd.z["uid"][a:b]
            out.setdefault("RCA_misses", []).append({
                "incident": iid, "fault_type": ftype, "target": tgt, "first_causal_rank": best_rank,
                "n_causal": int(causal.sum()),
                "top5": [f"{u['svc'][int(uid[i])]}: {u['masked'][int(uid[i])][:100]}" for i in order[:5]],
                "causal_examples": [f"{u['svc'][int(x)]}: {u['masked'][int(x)][:100]}" for x in uid[causal][:3]]})
    Path("results/error_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()}))
    _ = torch


if __name__ == "__main__":
    main()
