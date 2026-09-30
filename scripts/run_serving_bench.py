"""CPU serving benchmark matrix -> results/bench/*.json and docs/serving_benchmark.md.

Runs `loglens bench` on 1M-line BGL and HDFS raw log files for FP32 vs int8, cache on/off, and
several core counts (process tree pinned with cpu_affinity, so '4 cores' really is 4 cores).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CONFIGS = [
    # name, args
    ("int8 + cache, 4 cores (2 ORT threads, 3 mask workers)", ["--cores", "4", "--threads", "2", "--mask-workers", "3"]),
    ("int8 + cache, 4 cores, single process (4 ORT threads)", ["--cores", "4", "--threads", "4"]),
    ("int8 + cache, 8 cores (4 ORT threads, 6 mask workers)", ["--cores", "8", "--threads", "4", "--mask-workers", "6"]),
    ("int8 + cache, 1 core", ["--cores", "1", "--threads", "1"]),
    ("fp32 + cache, 4 cores (2 ORT threads, 3 mask workers)", ["--no-int8", "--cores", "4", "--threads", "2", "--mask-workers", "3"]),
    ("int8, NO cache, 4 cores (2 ORT threads, 3 mask workers)", ["--no-cache", "--cores", "4", "--threads", "2", "--mask-workers", "3"]),
]


def main() -> None:
    rows = []
    Path("results/bench").mkdir(parents=True, exist_ok=True)
    for ds in ("bgl", "hdfs"):
        f = f"data/bench/{ds}_1m.log"
        for i, (name, args) in enumerate(CONFIGS):
            out = subprocess.run([sys.executable, "-m", "loglens.serve.cli", "bench", f, "--model-dir",
                                  "artifacts/onnx", *args], capture_output=True, text=True,
                                 env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
            txt = out.stdout[out.stdout.index("{"):]
            r = json.loads(txt)
            Path(f"results/bench/{ds}_{i}.json").write_text(json.dumps({"name": name, **r}, indent=2))
            rows.append((ds, name, r))
            print(ds, name, f"{r['lines_per_s']:,.0f} lines/s", flush=True)
    md = ["# CPU serving benchmark", "",
          "`loglens bench` on the first 1,000,000 raw lines of BGL and HDFS (generic parser, no dataset "
          "specific header handling). Batches of 20,000 lines; the whole process tree is pinned to N cores "
          "with `cpu_affinity` (the machine has 16 logical cores; numbers are for the pinned subset). "
          "ONNX Runtime 1.30, dynamic int8 unless stated. Peak RSS is the process (workers excluded).", "",
          "| dataset | configuration | lines/s | cache hit rate | p50 / p99 batch (ms) | peak RSS (MB) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for ds, name, r in rows:
        md.append(f"| {ds.upper()} | {name} | {r['lines_per_s']:,.0f} | {r['cache_hit_rate'] * 100:.1f}% | "
                  f"{r['p50_batch_ms']:.0f} / {r['p99_batch_ms']:.0f} | {r['peak_rss_mb']:.0f} |")
    Path("docs/serving_benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
