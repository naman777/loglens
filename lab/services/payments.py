import os
import random
import time

from common import log
from fastapi import FastAPI

app = FastAPI()


@app.post("/charge")
def charge():
    pid = random.randint(100000, 999999)
    log("INFO", f"charge request payment_id={pid} amount=42.00 currency=USD")
    lat = int(os.environ.get("BANK_LATENCY_MS", "180"))
    time.sleep(lat / 1000)
    if lat > 1500:
        log("WARN", f"bank call slow: latency={lat}ms (downstream bank-sim degraded)")
    log("INFO", f"payment {pid} approved in {lat}ms")
    return {"payment_id": pid}


@app.post("/debug/hold")
def hold(n: int = 10):
    """DB-pool-exhaustion fault: hold connections open."""
    log("WARN", f"connection pool exhausted (in_use={n}/10), waiting for free connection")
    return {"held": n}
