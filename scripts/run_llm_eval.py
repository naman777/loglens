"""Run the LLM baseline(s) and write results/llm.json (merged into results/benchmark.md by
`python -m loglens.eval.report --render-only`).

usage: python scripts/run_llm_eval.py [--provider hf_local|anthropic] [--model NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loglens.eval.llm_eval import LLMConfig, Provider, eval_anomaly, eval_rca  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="hf_local")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-prompt-tokens", type=int, default=3500)
    ap.add_argument("--price-in", type=float, default=0.0)
    ap.add_argument("--price-out", type=float, default=0.0)
    ap.add_argument("--price-date", default="")
    a = ap.parse_args()
    cfg = LLMConfig(provider=a.provider, model=a.model, n_anomaly_windows=a.n,
                    max_prompt_tokens=a.max_prompt_tokens, price_in_per_mtok=a.price_in,
                    price_out_per_mtok=a.price_out, price_date=a.price_date)
    prov = Provider(cfg)
    t0 = time.time()
    out = {"model": a.model, "provider": a.provider}
    out["anomaly_BGL"] = eval_anomaly(cfg, prov, "BGL")
    out["rca_lab"] = eval_rca(cfg, prov)
    out["cost"] = prov.cost()
    out["wall_seconds"] = time.time() - t0
    Path("results").mkdir(exist_ok=True)
    f = Path("results/llm.json")
    allr = json.loads(f.read_text()) if f.exists() else {}
    allr[a.model] = out
    f.write_text(json.dumps(allr, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
