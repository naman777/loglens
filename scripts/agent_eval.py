"""Agent-side measurement without an LLM in the loop: what the on-call agent would be handed.

For each lab test incident, the agent's first call is ``rank_suspect_lines(service='all', ...)``.
We report (a) input size the agent must read with vs without LogLens (GPT-2 token counts as a
model-agnostic proxy), (b) whether the returned lines contain a causal line, and (c) whether the top
lines point at the correct root-cause service. Actual agent answer quality / latency would need an
LLM and is NOT measured here.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer  # noqa: E402

from loglens.agent.tool import LogLensTool, LogSource  # noqa: E402
from loglens.data.lab import is_causal_rule  # noqa: E402
from loglens.serve.runtime import RuntimeConfig  # noqa: E402


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def main() -> None:
    gpt2 = Tokenizer.from_file("artifacts/gpt2_tokenizer.json")
    tool = LogLensTool(LogSource("lab/out"), cfg=RuntimeConfig(model_dir="artifacts/onnx", threads=4))
    incs = [json.loads(x) for f in sorted(Path("lab/out").glob("*/incidents.jsonl"))
            for x in f.read_text().splitlines() if x]
    summary = json.loads(Path("data/lab/summary.json").read_text())
    test_runs = {i["run_id"] for i in
                 __import__("polars").read_parquet("data/lab/incidents.parquet").filter(
                     __import__("polars").col("split") == "test").to_dicts()}
    incs = [i for i in incs if i["run_id"] in test_runs]
    rows_all = tool.rows()
    res = []
    for inc in incs:
        lo, hi = inc["t_start"] - 30_000, inc["t_end"] + 60_000
        window = f"{iso(lo)}/{iso(hi)}"
        raw = [r for r in rows_all if lo <= r["ts"] <= hi]
        raw_tokens = sum(len(gpt2.encode(f"{r['service']} {r['level']} {r['message']}").ids) for r in raw)
        out = tool.rank_suspect_lines("all", window, top_k=15)
        top = out["top_lines"]
        tok = sum(len(gpt2.encode(f"{t['service']} {t['level']} {t['text']}").ids) for t in top)
        causal_hit = any(is_causal_rule({"service": t["service"], "level": t["level"], "message": t["text"],
                                         "ts": t["ts"]}, inc) for t in top)
        top5 = Counter(t["service"] for t in top[:5]).most_common(1)
        res.append({"incident": inc["incident_id"], "fault_type": inc["fault_type"],
                    "raw_lines": len(raw), "raw_tokens": raw_tokens, "returned_tokens": tok,
                    "causal_in_top15": causal_hit,
                    "top1_service_correct": bool(top) and top[0]["service"] == inc["target_service"],
                    "top5_majority_service_correct": bool(top5) and top5[0][0] == inc["target_service"]})
    n = len(res)
    agg = {
        "incidents": n,
        "mean_raw_lines": float(np.mean([r["raw_lines"] for r in res])),
        "mean_raw_tokens": float(np.mean([r["raw_tokens"] for r in res])),
        "mean_returned_tokens": float(np.mean([r["returned_tokens"] for r in res])),
        "token_reduction_x": float(np.mean([r["raw_tokens"] for r in res]) /
                                   max(np.mean([r["returned_tokens"] for r in res]), 1)),
        "causal_line_in_top15": sum(r["causal_in_top15"] for r in res) / n,
        "top1_service_correct": sum(r["top1_service_correct"] for r in res) / n,
        "top5_majority_service_correct": sum(r["top5_majority_service_correct"] for r in res) / n,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/agent_eval.json").write_text(json.dumps({"summary": agg, "per_incident": res}, indent=2))
    print(json.dumps(agg, indent=2))
    _ = summary


if __name__ == "__main__":
    main()
