"""Reversible real-lab fault contexts. The campaign runner verifies effects and labels.

Six faults are supported. Network netem, packet loss, CPU hog and memory leak remain
simulator-only until their injection and recovery can be verified reliably.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-p", "loglens-lab", "-f", str(ROOT / "lab/docker-compose.yml")]
TARGETS = {"container_kill": "orders", "redis_down": "inventory", "db_pool_exhaustion": "payments",
           "disk_full": "worker", "bad_config": "orders", "slow_downstream": "payments"}


def sh(*args: str, timeout: float = 120) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout).stdout


def request(service: str, path: str, method: str = "GET") -> dict:
    # Run from gateway so stopped/unhealthy target containers can still be probed.
    code = """import json, time, urllib.request, urllib.error
start = time.monotonic()
try:
    req = urllib.request.Request(URL, method=METHOD)
    with urllib.request.urlopen(req, timeout=5) as response:
        result = {'status': response.status, 'body': json.loads(response.read())}
except urllib.error.HTTPError as exc:
    result = {'status': exc.code, 'body': {}}
except (urllib.error.URLError, TimeoutError) as exc:
    result = {'status': 0, 'body': {}, 'error': type(exc).__name__}
result['seconds'] = time.monotonic() - start
print(json.dumps(result))
"""
    code = f"URL = {('http://' + service + ':8000' + path)!r}\nMETHOD = {method!r}\n" + code
    return json.loads(sh(*COMPOSE, "exec", "-T", "gateway", "python", "-c", code, timeout=15))


def require_ok(service: str, path: str, method: str = "GET") -> dict:
    result = request(service, path, method)
    if result["status"] != 200:
        raise RuntimeError(f"{service}{path} returned {result}")
    return result["body"]


@contextmanager
def environment_fault(service: str, environment: dict):
    # JSON is valid YAML; an override recreates the actual serving container.
    with tempfile.TemporaryDirectory(prefix="loglens-compose-") as folder:
        override = Path(folder) / "override.json"
        override.write_text(json.dumps({"services": {service: {"environment": environment}}}))
        try:
            sh(*COMPOSE, "-f", str(override), "up", "-d", "--no-deps", "--force-recreate", service)
            yield
        finally:
            sh(*COMPOSE, "up", "-d", "--no-deps", "--force-recreate", service)


@contextmanager
def inject(fault: str):
    if fault not in TARGETS:
        raise ValueError(f"Unsupported real-lab fault: {fault}")
    if fault == "bad_config":
        with environment_fault("orders", {"PORT": "abc"}):
            yield
    elif fault == "slow_downstream":
        with environment_fault("payments", {"BANK_LATENCY_MS": "3000"}):
            yield
    elif fault in ("container_kill", "redis_down"):
        service = "orders" if fault == "container_kill" else "redis"
        try:
            sh(*COMPOSE, "kill" if fault == "container_kill" else "stop", service)
            yield
        finally:
            sh(*COMPOSE, "up", "-d", "--no-deps", service)
    elif fault == "db_pool_exhaustion":
        try:
            body = require_ok("payments", "/debug/hold?n=10", "POST")
            if body.get("held") != 10:
                raise RuntimeError("Database hold did not acquire all connections")
            yield
        finally:
            require_ok("payments", "/debug/release", "POST")
    elif fault == "disk_full":
        # Refuse to fill anything except this lab's bounded tmpfs, and cap the write.
        code = """import errno, os
root = '/var/spool/outbox'
assert any(parts[1] == root and parts[2] == 'tmpfs' for parts in
           (line.split() for line in open('/proc/mounts'))), 'spool must be tmpfs'
stats = os.statvfs(root)
assert stats.f_blocks * stats.f_frsize <= 16 * 1024 * 1024, 'spool exceeds 16 MiB'
try:
    with open(root + '/fill', 'wb', buffering=0) as stream:
        for _ in range(17):
            stream.write(b'x' * (1024 * 1024))
except OSError as exc:
    if exc.errno != errno.ENOSPC:
        raise
else:
    raise RuntimeError('spool did not fill')
"""
        try:
            sh(*COMPOSE, "exec", "-T", "worker", "python", "-c", code)
            yield
        finally:
            sh(*COMPOSE, "exec", "-T", "worker", "python", "-c",
               "from pathlib import Path; Path('/var/spool/outbox/fill').unlink(missing_ok=True)")
