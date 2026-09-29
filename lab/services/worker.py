"""Consumes tasks and 'sends' emails by spooling to disk (disk-full fault target)."""
import os
import time

from common import log
from fastapi import FastAPI

app = FastAPI()


@app.on_event("startup")
def startup():
    log("INFO", "worker service starting, version 1.4.2")
    log("INFO", "worker listening on port 8004, ready to serve")


@app.post("/task")
def task(order_id: int = 0):
    try:
        path = os.path.join(os.environ.get("SPOOL_DIR", "/tmp"), f"mail-{order_id}-{time.time()}")
        with open(path, "w") as f:
            f.write("hello")
        log("INFO", f"sent email to user for order_id={order_id}")
    except OSError as e:
        log("ERROR", f"OSError: [Errno {e.errno}] No space left on device: '/var/spool/mail/outbox'")
    return {"ok": True}
