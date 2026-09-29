"""Run fault-injection campaigns on the simulated lab and write logs + incident labels.

Usage: python lab/run_campaign.py [--campaigns 10] [--incidents 18] [--seed 1337] [--out lab/out]

Each campaign: 5 min warm-up, then loop {normal 6-12 min, inject one fault for 2-5 min, 1 min
recovery}. Load varies in waves (low/normal/peak) so "normal" is not flat, and ~25% of gaps contain
a short benign blip (hard negative that is NOT an incident).
Outputs per campaign: <out>/<run_id>/logs.jsonl and incidents.jsonl (label records).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.sim.engine import FAULT_TARGETS, FAULT_TYPES, Fault, Sim  # noqa: E402


def make_rps(rng: random.Random, horizon: float):
    segs, t = [], 0.0
    while t < horizon:
        d = rng.uniform(300, 900)
        segs.append((t, t + d, rng.choice([0.6, 1.5, 3.0, 1.5])))
        t += d

    def rps(x: float) -> float:
        for a, b, r in segs:
            if a <= x < b:
                return r * (1 + 0.15 * (((x * 7919) % 100) / 100 - 0.5))
        return 1.5

    return rps


def plan(rng: random.Random, n_incidents: int, order: list[str]) -> tuple[list[Fault], float]:
    faults, t = [], 300.0  # warm-up
    for i in range(n_incidents):
        t += rng.uniform(360, 720)
        if rng.random() < 0.25:  # benign blip, not an incident
            blip = Fault("network_latency", rng.choice(["orders", "inventory"]),
                         t - rng.uniform(60, 200), 0.0, intensity=0.12, notes="noise")
            blip.t_end = blip.t_start + rng.uniform(8, 20)
            faults.append(blip)
        ft = order[i % len(order)]
        dur = rng.uniform(120, 300)
        f = Fault(ft, rng.choice(FAULT_TARGETS[ft]), t, t + dur,
                  intensity=rng.uniform(0.6, 1.4), notes=f"intensity={0:.2f}")
        f.notes = f"intensity={f.intensity:.2f}"
        faults.append(f)
        t += dur + 60
    return faults, t + 120


def run_one(seed: int, run_id: str, n_incidents: int, out: Path, day_offset: int) -> int:
    rng = random.Random(seed)
    order = FAULT_TYPES[:]
    rng.shuffle(order)
    faults, horizon = plan(rng, n_incidents, order)
    sim = Sim(seed)
    sim.faults = faults
    sim.run(0.0, horizon, make_rps(rng, horizon))
    base = datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp() + day_offset * 86400
    d = out / run_id
    d.mkdir(parents=True, exist_ok=True)
    lines = sorted(sim.lines, key=lambda x: x.ts)
    with open(d / "logs.jsonl", "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps({
                "ts": int((base + ln.ts) * 1000), "service": ln.service, "level": ln.level,
                "message": ln.message, "rid": ln.rid, "sim_fault": ln.fault,
            }) + "\n")
    with open(d / "incidents.jsonl", "w", encoding="utf-8") as f:
        k = 0
        for fa in faults:
            if fa.notes == "noise":
                continue
            fa.incident_id = f"{run_id}-i{k:02d}"
            k += 1
            f.write(json.dumps({
                "run_id": run_id, "incident_id": fa.incident_id, "fault_type": fa.fault_type,
                "target_service": fa.target, "t_start": int((base + fa.t_start) * 1000),
                "t_end": int((base + fa.t_end) * 1000), "notes": fa.notes,
            }) + "\n")
    return len(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaigns", type=int, default=10)
    ap.add_argument("--incidents", type=int, default=18)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="lab/out")
    a = ap.parse_args()
    for c in range(a.campaigns):
        run_id = f"c{c:02d}"
        n = run_one(a.seed * 100 + c, run_id, a.incidents, Path(a.out), c)
        print(f"{run_id}: {a.incidents} incidents, {n:,} lines")


if __name__ == "__main__":
    main()
