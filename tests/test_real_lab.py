"""Fault lifecycle and service semantics, without claiming real-container validation."""
import errno
import importlib
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from lab import run_real_campaign as campaign
from lab.faults import inject as faults


@pytest.fixture
def services(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "lab/services"))
    return SimpleNamespace(**{name: importlib.import_module(f"lab.services.{name}")
                              for name in ("payments", "orders", "inventory", "worker", "gateway")})


def test_pool_exhaustion_holds_connections_and_recovers(services, monkeypatch):
    import database

    monkeypatch.setenv("POOL_SIZE", "2")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    connections = []

    @contextmanager
    def connect(*args, **kwargs):
        conn = SimpleNamespace(closed=False, execute=lambda *a: None)
        connections.append(conn)
        try:
            yield conn
        finally:
            conn.closed = True

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))
    monkeypatch.setattr(services.payments, "db", database.Database())
    with TestClient(services.payments.app) as client:
        assert client.post("/debug/hold?n=2").json() == {"held": 2}
        assert not any(c.closed for c in connections)
        assert client.post("/charge").status_code == 503
        assert client.post("/debug/release").status_code == 200
        assert all(c.closed for c in connections)
        assert client.post("/charge").status_code == 200


def test_failed_connect_releases_capacity(monkeypatch, services):
    import database

    monkeypatch.setenv("POOL_SIZE", "1")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    pool = database.Database()

    def fail(*args, **kwargs):
        raise OSError("database offline")

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=fail))
    for _ in range(2):
        with pytest.raises(OSError):
            with pool.connection():
                pass


def test_orders_persist_before_enqueue_and_surface_queue_failure(services, monkeypatch):
    calls = []

    @contextmanager
    def connection():
        yield SimpleNamespace(execute=lambda sql, *args: calls.append(sql))

    def fail_queue(*args):
        assert any("INSERT INTO orders" in sql for sql in calls)
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr(services.orders, "db", SimpleNamespace(connection=connection))
    monkeypatch.setattr(services.orders, "queue", lambda: SimpleNamespace(lpush=fail_queue))
    assert TestClient(services.orders.app).post("/create").status_code == 503


def test_gateway_propagates_upstream_http_failure(services, monkeypatch):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(503, json={"detail": "failed"}))
    monkeypatch.setattr(services.gateway.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=transport, **kwargs))
    assert TestClient(services.gateway.app).post("/orders").status_code == 502


def test_worker_disk_failure_is_not_success(services, monkeypatch, tmp_path):
    monkeypatch.setenv("SPOOL_DIR", str(tmp_path))

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "open", full)
    assert TestClient(services.worker.app).post("/task?order_id=example").status_code == 507


def test_worker_retries_pending_task_before_acknowledgement(services, monkeypatch):
    import threading

    stop = threading.Event()
    attempts, ack = [], []
    cache = SimpleNamespace(lindex=lambda *a: "order1", lrem=lambda *a: ack.append(a))
    monkeypatch.setattr(services.worker.redis, "Redis", lambda **kw: cache)

    def spool(oid):
        attempts.append(oid)
        if len(attempts) == 1:
            raise HTTPException(507, "disk full")
        stop.set()

    monkeypatch.setattr(services.worker, "spool", spool)
    services.worker.consume(stop)
    assert attempts == ["order1", "order1"]
    assert ack == [("email_pending", 1, "order1")]


@pytest.mark.parametrize("fault", ["container_kill", "redis_down", "bad_config", "slow_downstream", "disk_full"])
def test_fault_recovers_on_exception(fault, monkeypatch):
    commands = []
    monkeypatch.setattr(faults, "sh", lambda *args, **kw: commands.append(args) or "")
    with pytest.raises(RuntimeError, match="probe failed"):
        with faults.inject(fault):
            raise RuntimeError("probe failed")
    assert len(commands) == 2
    assert "up" in commands[-1] or "unlink" in commands[-1][-1]
    if fault in ("bad_config", "slow_downstream"):
        assert "run" not in commands[0]
        assert "--force-recreate" in commands[0]
        assert "-e" not in commands[0]


def test_pool_fault_releases_even_when_verification_fails(monkeypatch):
    requests = []

    def call(service, path, method):
        requests.append((path, method))
        return {"held": 9}

    monkeypatch.setattr(faults, "require_ok", call)
    with pytest.raises(RuntimeError, match="all connections"):
        with faults.inject("db_pool_exhaustion"):
            pytest.fail("unverified injection was accepted")
    assert requests == [("/debug/hold?n=10", "POST"), ("/debug/release", "POST")]


def test_request_probe_code_is_executable(monkeypatch):
    def fake_sh(*args, **kwargs):
        compile(args[-1], "probe", "exec")
        assert "METHOD = 'POST'" in args[-1]
        return '{"status": 503, "body": {}}'

    monkeypatch.setattr(faults, "sh", fake_sh)
    assert faults.request("payments", "/charge", "POST")["status"] == 503


def test_subprocess_failures_are_not_swallowed(monkeypatch):
    def fail(*args, **kwargs):
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        faults.sh("docker", "info")


def test_no_incident_result_without_sustained_effect(monkeypatch):
    recovered = []

    @contextmanager
    def inject(fault):
        try:
            yield
        finally:
            recovered.append(True)

    monkeypatch.setattr(campaign, "inject", inject)
    monkeypatch.setattr(campaign, "wait_for", lambda *a, **kw: {"ok": True})
    monkeypatch.setattr(campaign, "effect", lambda *a: None)
    with pytest.raises(RuntimeError, match="did not persist"):
        campaign.run_incident("redis_down", 0)
    assert recovered == [True]


def test_no_incident_result_when_recovery_fails(monkeypatch):
    @contextmanager
    def inject(fault):
        yield

    calls = []

    def wait_for(*args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError("recovery failed")
        return {"ok": True}

    monkeypatch.setattr(campaign, "inject", inject)
    monkeypatch.setattr(campaign, "wait_for", wait_for)
    monkeypatch.setattr(campaign, "effect", lambda *a: {"status": 503})
    with pytest.raises(RuntimeError, match="recovery failed"):
        campaign.run_incident("redis_down", 0)


def test_log_capture_filters_noise_and_sorts(monkeypatch, tmp_path):
    rows = [{"ts": ts, "service": "orders", "level": "ERROR", "message": "failed"} for ts in (2, 1)]
    monkeypatch.setattr(campaign, "sh", lambda *a: "startup noise\n" + "\n".join(map(json.dumps, rows)))
    campaign.save_logs(tmp_path)
    actual = [json.loads(line) for line in (tmp_path / "logs.jsonl").read_text().splitlines()]
    assert [r["ts"] for r in actual] == [1, 2]
    assert "startup noise" in (tmp_path / "compose.log").read_text()
