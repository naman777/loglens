import os
import random

from common import log
from fastapi import FastAPI

app = FastAPI()
LEAK: list = []


@app.post("/create")
def create():
    oid = random.randint(100000, 999999)
    log("INFO", f"creating order order_id={oid}")
    log("INFO", f"order {oid} persisted, enqueue email task")
    return {"order_id": oid, "pool": os.environ.get("POOL_SIZE")}


@app.get("/debug/leak")
def leak(mb: int = 50):
    """memory-leak fault: allocate until OOM."""
    LEAK.append(bytearray(mb * 1024 * 1024))
    log("WARN", f"heap usage {len(LEAK) * mb}MB of 1024MB")
    return {"held_mb": len(LEAK) * mb}
