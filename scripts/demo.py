"""Terminal demo: simulate a fresh lab run, inject a fault, let LogLens find it.

    python scripts/demo.py [--fault db_pool_exhaustion] [--seed 7]

Uses the simulator in lab/sim (no Docker) and the int8 ONNX model in artifacts/onnx.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.run_campaign import make_rps  # noqa: E402
from lab.sim.engine import FAULT_TARGETS, FAULT_TYPES, Fault, Sim  # noqa: E402
from loglens.serve.runtime import LogLensRuntime, RuntimeConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", default="db_pool_exhaustion", choices=FAULT_TYPES)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    target = rng.choice(FAULT_TARGETS[a.fault])
    sim = Sim(a.seed)
    sim.faults = [Fault(a.fault, target, 420.0, 600.0, 1.0)]
    sim.run(0.0, 720.0, make_rps(rng, 720.0))
    lines = sorted(sim.lines, key=lambda x: x.ts)
    raw = [json.dumps({"ts": int(1.8e12 + ln.ts * 1000), "service": ln.service, "level": ln.level,
                       "message": ln.message}) for ln in lines]
    print(f"simulated {len(raw):,} log lines from 5 services over 12 minutes; "
          f"injecting {a.fault} on '{target}' at t=7:00 (kept secret from the model)\n")
    rt = LogLensRuntime(RuntimeConfig(model_dir="artifacts/onnx", threads=4))
    win = 600
    for t_end in range(120, 721, 120):
        chunk = [r for r, ln in zip(raw, lines, strict=True) if t_end - 120 <= ln.ts < t_end][-2000:]
        t0 = time.perf_counter()
        res = rt.score(chunk, top_k=5)
        ms = (time.perf_counter() - t0) * 1000
        flag = "ANOMALY" if res.anomaly > rt.cfg.threshold else "ok     "
        print(f"[{t_end // 60}:{t_end % 60:02d}] {flag} score={res.anomaly:.3f}  {len(chunk)} lines scored in {ms:.0f} ms")
        if res.anomaly > rt.cfg.threshold:
            for s in res.lines:
                d = json.loads(chunk[s["index"]])
                print(f"      {s['score']:>6.2f}  [{d['service']:<9}] {d['level']:<5} {d['message'][:100]}")
            break
    _ = win


if __name__ == "__main__":
    main()
