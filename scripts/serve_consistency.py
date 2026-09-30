"""Train/serve consistency: score RAW BGL/HDFS-style lines through the ONNX runtime and compare the
window-level F1 with the offline evaluation (same windows, same validation-tuned threshold).

usage: python scripts/serve_consistency.py [--format generic|bgl|hdfs]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loglens.eval.metrics import prf  # noqa: E402
from loglens.serve.runtime import LogLensRuntime, RuntimeConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="generic")
    ap.add_argument("--service", default="BGL")
    ap.add_argument("--n-windows", type=int, default=1500)
    a = ap.parse_args()
    res = json.loads(Path("results/results.json").read_text())["anomaly"]["BGL"]["LogLens-sup"]
    thr = 1 / (1 + np.exp(-res["threshold"]))
    lines_df = pl.read_parquet("data/masked/BGL/lines.parquet", columns=["line_no", "split", "label"])
    test = lines_df.filter(pl.col("split") == "test").sort("line_no")
    start, n = int(test["line_no"][0]), len(test)
    raw = []
    with open("data/raw/BGL/BGL.log", "rb") as f:
        for i, line in enumerate(f):
            if i >= start:
                raw.append(line.decode("utf-8", "replace").rstrip("\n"))
            if len(raw) >= n:
                break
    rt = LogLensRuntime(RuntimeConfig(model_dir="artifacts/onnx", threads=4, mask_workers=3,
                                      default_service=a.service if a.service != "none" else None,
                                      line_format=a.format))
    labels = np.array([0 if ln.startswith("- ") else 1 for ln in raw])
    W = 256
    starts = (np.linspace(0, len(raw) - W - 1, a.n_windows) // 128 * 128).astype(int)
    y, p = [], []
    for s in starts:
        r = rt.score(raw[s: s + W], top_k=1)
        y.append(int(labels[s: s + W].max()))
        p.append(r.anomaly)
    y, p = np.array(y), np.array(p)
    out = prf(y, p >= thr)
    from loglens.eval.metrics import pr_auc

    out["pr_auc"] = pr_auc(y, p)
    out["windows"] = len(y)
    out["positive_rate"] = float(y.mean())
    out["format"] = a.format
    out["service"] = a.service
    print(json.dumps(out, indent=2))
    Path("results/serve_consistency.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
