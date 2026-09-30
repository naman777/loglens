import os
import random

import redis
from common import log
from database import db
from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/stock")
def stock():
    sku = random.randint(1000, 9999)
    try:
        cache = redis.Redis(host=os.environ.get("REDIS_HOST", "redis"),
                            socket_connect_timeout=0.3, socket_timeout=0.5)
        value = cache.get(f"sku:{sku}")
        if value is not None:
            log("INFO", f"cache hit key=sku:{sku}")
            return {"sku": sku, "source": "redis", "stock": int(value)}
    except redis.RedisError:
        log("ERROR", f"redis unavailable: redis:6379 (key=sku:{sku})")
    try:
        with db.connection() as conn:
            conn.execute("SELECT 1").fetchone()
        log("INFO", f"postgres stock fallback key=sku:{sku}")
        return {"sku": sku, "source": "postgres", "stock": 1}
    except Exception as exc:
        log("ERROR", f"inventory database unavailable: {type(exc).__name__}")
        raise HTTPException(503, "inventory unavailable") from exc
