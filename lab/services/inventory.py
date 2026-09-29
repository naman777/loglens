import os
import random

from common import log
from fastapi import FastAPI

app = FastAPI()


@app.get("/stock")
def stock():
    sku = random.randint(1000, 9999)
    try:
        import redis

        redis.Redis(host=os.environ.get("REDIS_HOST", "redis"), socket_connect_timeout=0.3).get(f"sku:{sku}")
        log("INFO", f"cache hit key=sku:{sku}")
    except Exception:
        log("ERROR", f"redis connection refused: redis:6379 (key=sku:{sku})")
        log("WARN", "cache unavailable, falling back to postgres")
    return {"sku": sku}
