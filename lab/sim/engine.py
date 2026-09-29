"""Discrete-event simulator of the 5-service lab app, with fault injection and ground truth.

Docker is not required: this reproduces the *log behaviour* of the Compose lab
(``lab/docker-compose.yml``) deterministically and orders of magnitude faster. Each emitted line is
tagged internally with ``fault`` = True when it was produced *because of* the injected fault at the
target service (simulator ground truth, used to validate the rule-based causal labeller).

Services: gateway, orders, payments, inventory, worker (+ postgres/redis as dependencies).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

FAULT_TYPES = [
    "container_kill", "network_latency", "packet_loss", "db_pool_exhaustion", "redis_down",
    "disk_full", "bad_config", "memory_leak", "cpu_hog", "slow_downstream",
]

# faults and the services each can hit (target_service)
FAULT_TARGETS = {
    "container_kill": ["gateway", "orders", "payments", "inventory", "worker"],
    "network_latency": ["orders", "payments", "inventory"],
    "packet_loss": ["orders", "payments", "inventory"],
    "db_pool_exhaustion": ["orders", "payments"],
    "redis_down": ["inventory"],
    "disk_full": ["worker"],
    "bad_config": ["gateway", "orders", "payments", "inventory", "worker"],
    "memory_leak": ["orders", "inventory", "payments"],
    "cpu_hog": ["gateway", "orders", "payments", "inventory", "worker"],
    "slow_downstream": ["payments"],
}

# regex-free keyword lists used by the causal-line labeller (also read by the rule labeller)
FAULT_KEYWORDS = {
    "container_kill": ["starting", "listening", "ready", "restarted", "recovered"],
    "network_latency": ["slow", "latency", "timeout", "timed out"],
    "packet_loss": ["retransmit", "reset by peer", "retry", "retrying", "timeout", "packet"],
    "db_pool_exhaustion": ["pool", "acquire", "connection", "exhausted"],
    "redis_down": ["redis", "cache", "connection refused", "fallback"],
    "disk_full": ["no space", "disk", "spool", "errno 28"],
    "bad_config": ["config", "invalid", "exiting", "env"],
    "memory_leak": ["heap", "gc", "memory", "oom", "killed"],
    "cpu_hog": ["cpu", "load average", "event loop", "lag", "slow"],
    "slow_downstream": ["bank", "slow", "timeout", "downstream", "latency"],
}

SERVICES = ["gateway", "orders", "payments", "inventory", "worker"]
DEP = {"gateway": ["orders", "payments", "inventory"], "orders": ["postgres"],
       "payments": ["postgres", "bank"], "inventory": ["redis", "postgres"], "worker": ["redis"]}
PORT = {"gateway": 8000, "orders": 8001, "payments": 8002, "inventory": 8003, "worker": 8004}


@dataclass
class Fault:
    fault_type: str
    target: str
    t_start: float
    t_end: float
    intensity: float = 1.0  # 0.5 mild .. 1.5 severe
    notes: str = ""
    incident_id: str = ""

    def active(self, t: float) -> bool:
        return self.t_start <= t < self.t_end


@dataclass
class Line:
    ts: float
    service: str
    level: str
    message: str
    fault: bool = False  # simulator ground truth: emitted because of the injected fault
    rid: str = ""


@dataclass
class Sim:
    seed: int
    lines: list[Line] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.faults: list[Fault] = []
        self.pool_in_use = {"orders": 3, "payments": 3}
        self.heap_mb = {"orders": 180, "inventory": 160, "payments": 170}
        self._n = 0

    # ---------------------------------------------------------------- helpers
    def fault_on(self, t: float, service: str) -> Fault | None:
        for f in self.faults:
            if f.target == service and f.active(t):
                return f
        return None

    def emit(self, t: float, svc: str, level: str, msg: str, fault: bool = False, rid: str = "") -> None:
        self.lines.append(Line(t, svc, level, msg, fault, rid))

    def rid(self) -> str:
        self._n += 1
        return f"{self.rng.getrandbits(32):08x}"

    def lat(self, mean: float, sigma: float = 0.35) -> float:
        return max(1.0, self.rng.lognormvariate(math.log(mean), sigma))

    def dead(self, t: float, svc: str) -> bool:
        """Service is down (killed / crash-looping on bad config / OOMed)."""
        f = self.fault_on(t, svc)
        if not f:
            return False
        if f.fault_type == "container_kill":
            return t < f.t_start + 12 * f.intensity + 8  # restart takes ~10-25 s
        if f.fault_type == "bad_config":
            return True
        if f.fault_type == "memory_leak":
            span = f.t_end - f.t_start
            return t > f.t_start + 0.8 * span and int(t) % 40 < 25  # OOM crash loop near the end
        return False

    # ---------------------------------------------------------------- one call from a caller
    def call(self, t: float, caller: str, callee: str, rid: str, what: str) -> tuple[bool, float]:
        """Simulate ``caller`` calling ``callee``. Returns (ok, elapsed_seconds). Emits caller-side
        symptom lines (never flagged as fault lines: they live in a different service)."""
        rng = self.rng
        base = {"orders": 25, "payments": 60, "inventory": 15, "postgres": 6, "redis": 1.5,
                "bank": 180}.get(callee, 20)
        f = self.fault_on(t, callee)
        if callee in SERVICES and self.dead(t, callee):
            self.emit(t, caller, "ERROR",
                      f"connection refused calling {callee}:{PORT[callee]} for {what}", rid=rid)
            self.emit(t + 0.001, caller, "WARN",
                      f"upstream {callee} unavailable, giving up after 1 attempt", rid=rid)
            return False, 0.002
        if callee == "redis" and (rf := self.fault_on(t, "inventory")) and rf.fault_type == "redis_down":
            return False, 0.001
        if f is not None:
            ft = f.fault_type
            if ft == "network_latency" and callee in SERVICES:
                d = base * rng.uniform(40, 90) * f.intensity
                if d > 2000:
                    self.emit(t + 2.0, caller, "ERROR",
                              f"timeout after 2000ms calling {callee} for {what}", rid=rid)
                    return False, 2.0
                return True, d / 1000
            if ft == "cpu_hog" and callee in SERVICES:
                d = base * rng.uniform(8, 30) * f.intensity
                if d > 1500:
                    self.emit(t + 1.5, caller, "WARN", f"slow response from {callee}: {int(d)}ms", rid=rid)
                return True, d / 1000
            if ft == "packet_loss" and callee in SERVICES and rng.random() < 0.35 * f.intensity:
                self.emit(t + 0.2, caller, "WARN",
                          f"connection reset by peer calling {callee}, retrying (attempt 1/3)", rid=rid)
                if rng.random() < 0.4:
                    self.emit(t + 2.2, caller, "ERROR", f"timeout after 2000ms calling {callee}",
                              rid=rid)
                    return False, 2.2
                return True, 0.25 + base / 1000
            if ft == "slow_downstream" and callee == "bank":
                d = base * rng.uniform(9, 12) * f.intensity
                return d < 4500, min(d, 5000) / 1000
            if ft == "memory_leak" and callee in SERVICES:
                heap = self.heap_mb.get(callee, 200)
                return True, base / 1000 * (1 + heap / 300)
        return True, self.lat(base) / 1000

    # ---------------------------------------------------------------- service handlers
    def handle_browse(self, t: float, rid: str) -> float:
        rng = self.rng
        sku = rng.randint(1000, 9999)
        ok, el = self.call(t, "gateway", "inventory", rid, "GET /products")
        if not ok:
            return el
        t1 = t + el * 0.3
        self.inventory_lookup(t1, rid, sku)
        return el

    def inventory_lookup(self, t: float, rid: str, sku: int) -> None:
        rng = self.rng
        rf = self.fault_on(t, "inventory")
        redis_down = rf is not None and rf.fault_type == "redis_down"
        if redis_down:
            self.emit(t, "inventory", "ERROR",
                      f"redis connection refused: redis:6379 (key=sku:{sku})", True, rid)
            self.emit(t + 0.001, "inventory", "WARN",
                      "cache unavailable, falling back to postgres", True, rid)
            q = self.lat(90 * rf.intensity, 0.5)
            self.emit(t + q / 1000, "inventory", "WARN",
                      f"slow query select_stock took {int(q)}ms rows=1", True, rid)
            return
        if rng.random() < 0.82:
            self.emit(t, "inventory", "INFO", f"cache hit key=sku:{sku}", rid=rid)
        else:
            self.emit(t, "inventory", "INFO", f"cache miss key=sku:{sku}, querying postgres", rid=rid)
            q = self.lat(8)
            self.emit(t + q / 1000, "inventory", "INFO",
                      f"inventory query ok rows=1 in {int(q)}ms", rid=rid)
        self.service_degradation(t, "inventory", rid)

    def service_degradation(self, t: float, svc: str, rid: str) -> None:
        """Fault-driven target-service lines that fire on normal request handling."""
        f = self.fault_on(t, svc)
        if not f:
            return
        rng = self.rng
        ft = f.fault_type
        if ft == "network_latency" and rng.random() < 0.6:
            self.emit(t, svc, "WARN",
                      f"slow response to client: latency={int(self.lat(1800 * f.intensity))}ms "
                      "(tc netem delay suspected)", True, rid)
        elif ft == "packet_loss" and rng.random() < 0.5:
            self.emit(t, svc, "WARN",
                      f"tcp retransmit detected on inbound connection, peer={rng.choice(SERVICES)}",
                      True, rid)
        elif ft == "cpu_hog":
            if rng.random() < 0.5:
                self.emit(t, svc, "WARN",
                          f"high cpu load average={rng.uniform(6, 14):.1f}, event loop lag "
                          f"{int(self.lat(700 * f.intensity))}ms", True, rid)
        elif ft == "memory_leak":
            span = max(f.t_end - f.t_start, 1)
            self.heap_mb[svc] = 180 + 800 * min((t - f.t_start) / (0.8 * span), 1.0)
            heap = int(self.heap_mb[svc])
            if rng.random() < 0.35:
                self.emit(t, svc, "WARN" if heap > 450 else "INFO",
                          f"heap usage {heap}MB of 1024MB" + (", GC pause "
                          f"{int(self.lat(heap / 2))}ms" if heap > 500 else ""), heap > 450, rid)
        else:
            if svc in self.heap_mb:
                self.heap_mb[svc] = 180

    def handle_order(self, t: float, rid: str) -> float:
        rng = self.rng
        ok, el = self.call(t, "gateway", "orders", rid, "POST /orders")
        if not ok:
            return el
        oid = rng.randint(100000, 999999)
        user = rng.randint(1, 5000)
        self.emit(t, "orders", "INFO", f"creating order order_id={oid} user_id={user} items={rng.randint(1, 5)}", rid=rid)
        self.db_op(t + 0.002, "orders", rid, f"INSERT order {oid}")
        self.emit(t + 0.02, "orders", "INFO", f"order {oid} persisted, enqueue email task", rid=rid)
        self.service_degradation(t, "orders", rid)
        # worker
        self.emit(t + rng.uniform(0.05, 1.0), "worker", "INFO", f"picked task send_email order_id={oid}", rid=rid)
        self.worker_task(t + 0.5, rid, oid)
        return el + 0.03

    def worker_task(self, t: float, rid: str, oid: int) -> None:
        rng = self.rng
        if self.dead(t, "worker"):
            return
        f = self.fault_on(t, "worker")
        if f and f.fault_type == "disk_full":
            self.emit(t, "worker", "ERROR",
                      "OSError: [Errno 28] No space left on device: '/var/spool/mail/outbox'", True, rid)
            self.emit(t + 0.01, "worker", "ERROR",
                      f"failed to write spool file for order_id={oid}, will retry", True, rid)
            if rng.random() < 0.3:
                self.emit(t + 0.5, "worker", "WARN", "disk usage 100% on /var/spool, 0 bytes free", True, rid)
            return
        self.service_degradation(t, "worker", rid)
        self.emit(t + rng.uniform(0.05, 0.4), "worker", "INFO", f"sent email to user for order_id={oid}", rid=rid)

    def db_op(self, t: float, svc: str, rid: str, what: str) -> float:
        rng = self.rng
        f = self.fault_on(t, svc)
        if f and f.fault_type == "db_pool_exhaustion":
            hold = min(10, self.pool_in_use[svc] + rng.randint(2, 4))
            self.pool_in_use[svc] = hold
            self.emit(t, svc, "WARN",
                      f"connection pool exhausted (in_use={hold}/10), waiting for free connection", True, rid)
            if rng.random() < 0.6 * f.intensity:
                self.emit(t + 5.0, svc, "ERROR",
                          f"timeout acquiring db connection after 5000ms (QueuePool limit of size 10 reached) for {what}",
                          True, rid)
                return 5.0
        else:
            self.pool_in_use[svc] = rng.randint(1, 5)
        q = self.lat(7)
        if rng.random() < 0.03:
            q *= 30
            self.emit(t + q / 1000, svc, "WARN", f"slow query {what} took {int(q)}ms", rid=rid)
        else:
            self.emit(t + q / 1000, svc, "INFO", f"db ok: {what} in {int(q)}ms pool_in_use={self.pool_in_use[svc]}/10", rid=rid)
        return q / 1000

    def handle_pay(self, t: float, rid: str) -> float:
        rng = self.rng
        ok, el = self.call(t, "gateway", "payments", rid, "POST /pay")
        if not ok:
            return el
        amt = rng.randint(5, 500)
        pid = rng.randint(100000, 999999)
        self.emit(t, "payments", "INFO", f"charge request payment_id={pid} amount={amt}.00 currency=USD", rid=rid)
        el2 = self.db_op(t + 0.003, "payments", rid, f"SELECT account for payment {pid}")
        t2 = t + el2
        bok, bel = self.call(t2, "payments", "bank", rid, f"charge {pid}")
        f = self.fault_on(t2, "payments")
        if f and f.fault_type == "slow_downstream":
            self.emit(t2 + bel, "payments", "WARN",
                      f"bank call slow: latency={int(bel * 1000)}ms (downstream bank-sim degraded)", True, rid)
            if not bok:
                self.emit(t2 + bel, "payments", "ERROR",
                          f"bank timeout after 5000ms for payment {pid}, downstream slow", True, rid)
                self.emit(t2 + bel + 0.01, "gateway", "ERROR", f"payment failed status=504 payment_id={pid}", rid=rid)
                return el + bel
        if rng.random() < 0.05:
            self.emit(t2 + bel, "payments", "WARN", f"bank latency high {int(bel * 1000 + 300)}ms, retrying (attempt 1/3)", rid=rid)
        if rng.random() < 0.02:
            self.emit(t2 + bel, "payments", "ERROR", f"payment {pid} declined: insufficient funds", rid=rid)
        else:
            self.emit(t2 + bel, "payments", "INFO", f"payment {pid} approved in {int(bel * 1000)}ms", rid=rid)
        self.service_degradation(t, "payments", rid)
        return el + el2 + bel

    # ---------------------------------------------------------------- per request
    def request(self, t: float, kind: str) -> None:
        rng = self.rng
        rid = self.rid()
        if self.dead(t, "gateway"):
            return
        path = {"browse": "/products", "order": "/orders", "pay": "/pay"}[kind]
        method = "GET" if kind == "browse" else "POST"
        self.emit(t, "gateway", "INFO", f"request received method={method} path={path} rid={rid}", rid=rid)
        el = {"browse": self.handle_browse, "order": self.handle_order, "pay": self.handle_pay}[kind](t, rid)
        status = 200
        gf = self.fault_on(t, "gateway")
        if el > 1.9:
            status = 504
        elif el < 0.004 and kind != "browse":
            status = 503
        if gf and gf.fault_type == "cpu_hog":
            el += self.lat(600 * gf.intensity) / 1000
            if rng.random() < 0.5:
                self.emit(t + el, "gateway", "WARN",
                          f"high cpu load average={rng.uniform(6, 14):.1f}, event loop lag {int(el * 1000)}ms", True, rid)
        if gf and gf.fault_type == "network_latency":
            el += 1.5 * gf.intensity
            self.emit(t + el, "gateway", "WARN", f"slow response to client: latency={int(el * 1000)}ms", True, rid)
        lvl = "INFO" if status == 200 else "ERROR"
        self.emit(t + el, "gateway", lvl, f"request completed status={status} latency_ms={int(el * 1000)} rid={rid}", rid=rid)

    # ---------------------------------------------------------------- lifecycle lines
    def lifecycle(self, f: Fault) -> None:
        """Start/stop-time lines that only exist because of the fault."""
        rng = self.rng
        s, ft, tgt = f.t_start, f.fault_type, f.target
        if ft == "container_kill":
            up = f.t_start + 12 * f.intensity + 8
            self.emit(up, tgt, "INFO", f"{tgt} service starting, version 1.4.2", True)
            self.emit(up + 0.5, tgt, "INFO", f"connected to dependencies ({', '.join(DEP[tgt])})", True)
            self.emit(up + 1.0, tgt, "INFO", f"{tgt} listening on port {PORT[tgt]}, ready to serve", True)
        elif ft == "bad_config":
            bad = rng.choice(["BANK_URL", "DB_POOL_SIZE", "REDIS_HOST", "LOG_LEVEL", "PORT"])
            t = s
            while t < f.t_end:
                self.emit(t, tgt, "INFO", f"{tgt} service starting, version 1.4.3", True)
                self.emit(t + 0.4, tgt, "ERROR", f"config error: invalid value for env var {bad}='{rng.choice(['-1', 'abc', '', 'null'])}'", True)
                self.emit(t + 0.5, tgt, "ERROR", "failed to parse configuration, exiting with code 1", True)
                t += rng.uniform(7, 15)
            self.emit(f.t_end + 1, tgt, "INFO", f"{tgt} service starting, version 1.4.2", True)
            self.emit(f.t_end + 2, tgt, "INFO", f"{tgt} listening on port {PORT[tgt]}, ready to serve", True)
        elif ft == "memory_leak":
            span = f.t_end - f.t_start
            self.emit(f.t_start + 0.8 * span, tgt, "ERROR", "memory limit exceeded (1024MB), container OOMKilled", True)
            self.emit(f.t_end + 1, tgt, "INFO", f"{tgt} service starting, version 1.4.2", True)
        elif ft == "disk_full":
            for k in range(0, int(f.t_end - f.t_start), 20):
                self.emit(s + k, tgt, "WARN", f"disk usage {min(100, 96 + k // 20)}% on /var/spool, low free space", True)
        elif ft == "container_kill" or ft == "redis_down":
            pass
        if ft == "redis_down":
            self.emit(f.t_end + 1, "inventory", "INFO", "redis connection recovered, cache enabled", True)

    # ---------------------------------------------------------------- noise (non-incident) events
    def background(self, t0: float, t1: float) -> None:
        rng = self.rng
        t = t0
        while t < t1:
            t += rng.expovariate(1 / 45)
            svc = rng.choice(SERVICES)
            msg = rng.choice([
                (f"health check ok uptime={int(t - t0)}s", "INFO"),
                (f"gc pause {rng.randint(3, 40)}ms", "INFO"),
                ("config reload requested, no changes", "INFO"),
                (f"slow query background_vacuum took {rng.randint(120, 400)}ms", "WARN"),
                ("retrying connection to postgres (attempt 1/3)", "WARN"),
                (f"deprecated api version header v1 seen from client {rng.randint(1, 9)}", "WARN"),
            ])
            if not self.dead(t, svc):
                self.emit(t, svc, msg[1], msg[0])

    def run(self, t0: float, t1: float, rps_fn) -> None:
        """Generate traffic for [t0, t1) with request rate ``rps_fn(t)``."""
        rng = self.rng
        t = t0
        while t < t1:
            r = max(rps_fn(t), 0.05)
            t += rng.expovariate(r)
            k = rng.random()
            self.request(t, "browse" if k < 0.60 else "order" if k < 0.85 else "pay")
        self.background(t0, t1)
        for f in self.faults:
            if t0 <= f.t_start < t1:
                self.lifecycle(f)
