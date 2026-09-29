"""Shared JSON-to-stdout logging: one JSON object per line with a `service` field."""
import json
import os
import sys
import time

SERVICE = os.environ.get("SERVICE", "svc")


def log(level: str, message: str, **extra) -> None:
    rec = {"ts": int(time.time() * 1000), "service": SERVICE, "level": level, "message": message, **extra}
    sys.stdout.write(json.dumps(rec) + "\n")
    sys.stdout.flush()
