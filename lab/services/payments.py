import os
import time
import uuid
from contextlib import ExitStack, asynccontextmanager
from threading import Lock

from common import log
from database import db
from fastapi import FastAPI, HTTPException, Query

held = ExitStack()
held_count = 0
held_lock = Lock()


def release_connections():
    global held_count
    with held_lock:
        held.close()
        held_count = 0
    log("INFO", "held database connections released")
    return {"held": 0}


@asynccontextmanager
async def lifespan(app):
    yield
    release_connections()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/charge")
def charge():
    pid = uuid.uuid4().hex
    try:
        with db.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS payments (id TEXT PRIMARY KEY)")
            lat = int(os.environ.get("BANK_LATENCY_MS", "180"))
            time.sleep(lat / 1000)
            if lat > 1500:
                log("WARN", f"bank call slow: latency={lat}ms (downstream bank-sim degraded)")
            conn.execute("INSERT INTO payments (id) VALUES (%s)", (pid,))
    except HTTPException:
        raise
    except Exception as exc:
        log("ERROR", f"payment database operation failed: {type(exc).__name__}")
        raise HTTPException(503, "payment database unavailable") from exc
    log("INFO", f"payment {pid} approved in {lat}ms")
    return {"payment_id": pid}


@app.post("/debug/hold")
def hold(n: int = Query(default=10, ge=1, le=100)):
    global held_count
    with held_lock:
        if held_count:
            raise HTTPException(409, "connections already held")
        if n > db.size:
            raise HTTPException(400, "hold exceeds pool capacity")
        try:
            for _ in range(n):
                conn = held.enter_context(db.connection())
                conn.execute("SELECT 1")
                held_count += 1
        except Exception:
            held.close()
            held_count = 0
            raise
    log("WARN", f"holding database connections in_use={held_count}/{db.size}")
    return {"held": held_count}


@app.post("/debug/release")
def release():
    return release_connections()
