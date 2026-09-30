"""Consume Redis tasks; acknowledge only after successful bounded-disk spooling."""
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import redis
from common import log
from fastapi import FastAPI, HTTPException


def spool(order_id: str):
    if not order_id.isalnum() or len(order_id) > 64:
        raise HTTPException(400, "invalid order ID")
    path = Path(os.environ.get("SPOOL_DIR", "/tmp")) / f"mail-{order_id}"
    try:
        with path.open("w") as stream:
            stream.write("hello")
    except OSError as exc:
        # A failed write may have created an empty file: it is not a completed task.
        path.unlink(missing_ok=True)
        log("ERROR", f"spool failed errno={exc.errno}: {exc.strerror}")
        raise HTTPException(507, "spool write failed") from exc
    log("INFO", f"sent email to user for order_id={order_id}")


def consume(stop):
    cache = redis.Redis(host=os.environ.get("REDIS_HOST", "redis"), decode_responses=True,
                        socket_connect_timeout=0.5, socket_timeout=1)
    while not stop.is_set():
        try:
            # This lab runs one consumer. Keep failed work pending across restarts.
            oid = cache.lindex("email_pending", -1)
            if oid is None:
                oid = cache.rpoplpush("email_tasks", "email_pending")
            if oid is not None:
                spool(oid)
                cache.lrem("email_pending", 1, oid)
        except (redis.RedisError, HTTPException) as exc:
            log("ERROR", f"worker task pending: {type(exc).__name__}")
        stop.wait(0.2)


@asynccontextmanager
async def lifespan(app):
    stop = threading.Event()
    thread = threading.Thread(target=consume, args=(stop,), daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=3)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/completed/{order_id}")
def completed(order_id: str):
    if not order_id.isalnum() or len(order_id) > 64:
        raise HTTPException(400, "invalid order ID")
    path = Path(os.environ.get("SPOOL_DIR", "/tmp")) / f"mail-{order_id}"
    return {"completed": path.is_file() and path.stat().st_size > 0}


@app.post("/task")
def task(order_id: str):
    spool(order_id)
    return {"ok": True}
