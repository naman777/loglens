"""Check saved CPU inference artifacts without training, downloads, or an API key."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(model_dir: Path, tokenizer: Path, fp32: bool = False) -> dict:
    suffix = ".onnx" if fp32 else ".int8.onnx"
    required = [model_dir / (name + suffix) for name in ("encoder", "window")]
    required.append(tokenizer)
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing or empty artifacts: " + ", ".join(missing)
                                + ". Reproduce training/export or obtain the matching release weights.")
    from loglens.serve.runtime import LogLensRuntime, RuntimeConfig

    rt = LogLensRuntime(RuntimeConfig(model_dir=str(model_dir), tokenizer=str(tokenizer),
                                     int8=not fp32, threads=1))
    lines = [json.dumps({"ts": 1800000000000 + i * 1000, "service": "payments",
                         "level": "ERROR" if i == 150 else "INFO",
                         "message": "bank timeout after 5000ms" if i == 150 else "payment approved"})
             for i in range(300)]
    first = rt.score(lines, top_k=5)
    before = rt.cache.hit_rate
    second = rt.score(lines, top_k=5)
    if not (first.n_lines == len(lines) and len(first.lines) == 5
            and math.isfinite(first.anomaly) and 0 <= first.anomaly <= 1):
        raise RuntimeError("Invalid model response")
    if any(not math.isfinite(s["score"]) or not 0 <= s["index"] < len(lines)
           for s in first.lines):
        raise RuntimeError("Invalid ranked line scores or indices")
    if abs(first.anomaly - second.anomaly) > 1e-6 or rt.cache.hit_rate <= before:
        raise RuntimeError("Repeated inference/cache check failed")
    return {"status": "ok", "precision": "fp32" if fp32 else "int8", "lines": first.n_lines,
            "anomaly_score": first.anomaly, "cache_hit_rate": rt.cache.hit_rate,
            "note": "Synthetic smoke input; this is not an accuracy benchmark."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/onnx")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizer/loglens-bpe-16k.json")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    try:
        result = check(args.model_dir, args.tokenizer, args.fp32)
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        ap.exit(1, f"Smoke check failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
