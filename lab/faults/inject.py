"""Fault injector for the Docker lab: one function per fault type; each returns a label record.

Requires docker (and pumba for netem faults). The simulator in lab/sim mirrors these fault types.
"""
from __future__ import annotations

import subprocess
import time
import uuid

COMPOSE = ["docker", "compose", "-f", "lab/docker-compose.yml"]
PY_GET = "import urllib.request;urllib.request.urlopen('http://localhost:8000{}')"


def sh(*args: str) -> None:
    subprocess.run(args, check=False)


def _rec(fault: str, target: str, t0: float, notes: str = "") -> dict:
    return {"incident_id": uuid.uuid4().hex[:8], "fault_type": fault, "target_service": target,
            "t_start": int(t0 * 1000), "t_end": int(time.time() * 1000), "notes": notes}


def container_kill(svc: str, duration: float = 30) -> dict:
    t0 = time.time()
    sh("docker", "kill", f"lab-{svc}-1")
    time.sleep(duration)
    sh(*COMPOSE, "up", "-d", svc)
    return _rec("container_kill", svc, t0)


def network_latency(svc: str, ms: int = 2000, duration: float = 180) -> dict:
    t0 = time.time()
    sh("pumba", "netem", "--duration", f"{int(duration)}s", "delay", "--time", str(ms), f"lab-{svc}-1")
    return _rec("network_latency", svc, t0, f"delay={ms}ms")


def packet_loss(svc: str, pct: int = 30, duration: float = 180) -> dict:
    t0 = time.time()
    sh("pumba", "netem", "--duration", f"{int(duration)}s", "loss", "--percent", str(pct), f"lab-{svc}-1")
    return _rec("packet_loss", svc, t0, f"loss={pct}%")


def db_pool_exhaustion(svc: str = "payments", duration: float = 180) -> dict:
    t0 = time.time()
    sh("docker", "exec", f"lab-{svc}-1", "python", "-c", PY_GET.format("/debug/hold?n=10"))
    time.sleep(duration)
    return _rec("db_pool_exhaustion", svc, t0)


def redis_down(duration: float = 150) -> dict:
    t0 = time.time()
    sh(*COMPOSE, "stop", "redis")
    time.sleep(duration)
    sh(*COMPOSE, "start", "redis")
    return _rec("redis_down", "inventory", t0)


def disk_full(duration: float = 150) -> dict:
    t0 = time.time()
    sh("docker", "exec", "lab-worker-1", "sh", "-c", "dd if=/dev/zero of=/var/spool/outbox/fill bs=1M || true")
    time.sleep(duration)
    sh("docker", "exec", "lab-worker-1", "rm", "-f", "/var/spool/outbox/fill")
    return _rec("disk_full", "worker", t0)


def bad_config(svc: str, duration: float = 120) -> dict:
    t0 = time.time()
    sh(*COMPOSE, "run", "-d", "-e", "PORT=abc", svc)
    time.sleep(duration)
    sh(*COMPOSE, "up", "-d", "--force-recreate", svc)
    return _rec("bad_config", svc, t0)


def memory_leak(svc: str = "orders", duration: float = 180) -> dict:
    t0 = time.time()
    for _ in range(int(duration // 5)):
        sh("docker", "exec", f"lab-{svc}-1", "python", "-c", PY_GET.format("/debug/leak?mb=50"))
        time.sleep(5)
    return _rec("memory_leak", svc, t0)


def cpu_hog(svc: str, duration: float = 150) -> dict:
    t0 = time.time()
    sh("docker", "exec", "-d", f"lab-{svc}-1", "sh", "-c",
       f"timeout {int(duration)} sh -c 'while :; do :; done'")
    time.sleep(duration)
    return _rec("cpu_hog", svc, t0)


def slow_downstream(duration: float = 180) -> dict:
    t0 = time.time()
    sh(*COMPOSE, "up", "-d", "-e", "BANK_LATENCY_MS=1800", "payments")
    time.sleep(duration)
    sh(*COMPOSE, "up", "-d", "--force-recreate", "payments")
    return _rec("slow_downstream", "payments", t0)


FAULTS = {f.__name__: f for f in (container_kill, network_latency, packet_loss, db_pool_exhaustion,
                                  redis_down, disk_full, bad_config, memory_leak, cpu_hog,
                                  slow_downstream)}
