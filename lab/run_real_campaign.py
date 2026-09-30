"""Verify real Docker incidents; outputs stay separate from synthetic training data."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.faults.inject import COMPOSE, TARGETS, inject, request, require_ok, sh  # noqa: E402


def wait_for(check, timeout=60):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except (RuntimeError, subprocess.SubprocessError) as exc:
            last = exc
        time.sleep(0.5)
    raise RuntimeError(f"Readiness/effect check timed out: {last}")


def baseline():
    for service in ("orders", "payments", "inventory", "worker", "gateway"):
        require_ok(service, "/health")
    require_ok("gateway", "/products")
    require_ok("gateway", "/pay", "POST")
    order = require_ok("gateway", "/orders", "POST")
    wait_for(lambda: require_ok("worker", f"/completed/{order['order_id']}")["completed"], 15)
    return {"ok": True, "persisted_and_consumed_order": order["order_id"]}


def state(service):
    container = sh(*COMPOSE, "ps", "-a", "-q", service).strip()
    if not container:
        raise RuntimeError(f"No container for {service}")
    return json.loads(sh("docker", "inspect", "--format", "{{json .State}}", container))


def effect(fault):
    if fault in ("container_kill", "bad_config", "redis_down"):
        target = "redis" if fault == "redis_down" else "orders"
        current = state(target)
        if current["Running"]:
            return None
        # Invalid startup configuration must actually fail, not just be briefly stopped.
        if fault == "bad_config" and current.get("ExitCode", 0) == 0:
            return None
        probe = request("gateway", "/orders", "POST")
        if probe["status"] not in (502, 503, 504):
            return None
        return {"container_state": current, "request": probe}
    if fault == "db_pool_exhaustion":
        probe = request("payments", "/charge", "POST")
        return probe if probe["status"] == 503 else None
    if fault == "disk_full":
        probe = request("worker", "/task?order_id=faultprobe", "POST")
        return probe if probe["status"] == 507 else None
    if fault == "slow_downstream":
        require_ok("payments", "/health")
        probe = request("payments", "/charge", "POST")
        gateway = request("gateway", "/pay", "POST")
        if probe["status"] == 200 and probe["seconds"] >= 2.5 and gateway["status"] == 504:
            return {"direct": probe, "gateway": gateway}
    return None


def collect_traffic(stop, path):
    with path.open("a", encoding="utf-8") as stream:
        while not stop.is_set():
            for endpoint, method in (("/products", "GET"), ("/orders", "POST"), ("/pay", "POST")):
                if stop.is_set():
                    break
                try:
                    result = request("gateway", endpoint, method)
                except Exception as exc:
                    result = {"probe_error": str(exc)}
                stream.write(json.dumps({"ts": int(time.time() * 1000), "path": endpoint, **result}) + "\n")
                stream.flush()
            stop.wait(0.5)


def run_incident(fault, duration):
    before = wait_for(baseline)
    start = int(time.time() * 1000)
    with inject(fault):
        observed = wait_for(lambda: effect(fault), timeout=30)
        time.sleep(duration)
        # Confirm the effect persisted, rather than recording a transient startup failure.
        sustained = effect(fault)
        if not sustained:
            raise RuntimeError(f"{fault}: effect did not persist")
        end = int(time.time() * 1000)
    recovered = wait_for(baseline)
    return {"incident_id": uuid.uuid4().hex[:12], "fault_type": fault,
            "target_service": TARGETS[fault], "t_start": start, "t_end": end,
            "recovered_at": int(time.time() * 1000), "source": "docker",
            "verified": True, "causal_labels_reviewed": False,
            "evidence": {"baseline": before, "fault": observed,
                         "sustained": sustained, "recovery": recovered}}


def save_logs(out):
    raw = sh(*COMPOSE, "logs", "--no-color", "--no-log-prefix")
    (out / "compose.log").write_text(raw, encoding="utf-8")
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and {"ts", "service", "level", "message"} <= row.keys():
            rows.append(row)
    rows.sort(key=lambda row: row["ts"])
    (out / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--faults", nargs="+", choices=sorted(TARGETS), default=list(TARGETS))
    ap.add_argument("--duration", type=float, default=10)
    ap.add_argument("--out", type=Path, default=Path("lab/real-out"))
    ap.add_argument("--keep-up", action="store_true", help="leave this campaign's containers running")
    args = ap.parse_args()
    if args.duration <= 0:
        ap.error("--duration must be positive")
    try:
        sh("docker", "info", timeout=15)
        sh(*COMPOSE, "config", "--quiet")
        if sh(*COMPOSE, "ps", "-a", "-q").strip():
            ap.error("loglens-lab already has containers; stop/remove that lab before a fresh campaign")
    except (OSError, subprocess.SubprocessError) as exc:
        ap.exit(1, f"Docker preflight failed: {exc}\n")
    out = args.out / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    thread = None
    try:
        sh(*COMPOSE, "up", "-d", "--build", "--wait", "--wait-timeout", "120", timeout=600)
        wait_for(baseline)
        thread = threading.Thread(target=collect_traffic, args=(stop, out / "traffic.jsonl"), daemon=True)
        thread.start()
        with (out / "incidents.jsonl").open("w", encoding="utf-8") as stream:
            for fault in args.faults:
                record = run_incident(fault, args.duration)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(f"Verified {fault}: normal -> observed fault -> recovery", flush=True)
    except BaseException as exc:
        (out / "failure.json").write_text(json.dumps({"error": str(exc), "type": type(exc).__name__}), encoding="utf-8")
        raise
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=20)
        try:
            save_logs(out)
        finally:
            if not args.keep_up:
                sh(*COMPOSE, "down", "--volumes")
        print(f"Evidence saved in {out}. Causal labels still require manual review.", flush=True)


if __name__ == "__main__":
    main()
