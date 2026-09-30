import os
import uuid

import redis
from common import log
from database import db
from fastapi import FastAPI, HTTPException

app = FastAPI()


def queue():
    return redis.Redis(host=os.environ.get("REDIS_HOST", "redis"),
                       socket_connect_timeout=0.5, socket_timeout=1)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/create")
def create():
    oid = uuid.uuid4().hex
    try:
        with db.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO orders (id) VALUES (%s)", (oid,))
        queue().lpush("email_tasks", oid)
    except HTTPException:
        raise
    except Exception as exc:
        log("ERROR", f"order persistence/enqueue failed: {type(exc).__name__} order_id={oid}")
        raise HTTPException(503, "order dependency unavailable") from exc
    log("INFO", f"order {oid} persisted, enqueued email task")
    return {"order_id": oid}
