"""Bound concurrent real Postgres connections in each lab service."""
import os
from contextlib import contextmanager
from threading import BoundedSemaphore

from common import log
from fastapi import HTTPException


class Database:
    def __init__(self):
        self.size = int(os.environ.get("POOL_SIZE", "10"))
        self.slots = BoundedSemaphore(self.size)

    @contextmanager
    def connection(self):
        if not self.slots.acquire(timeout=0.25):
            log("ERROR", "connection pool exhausted, waiting for free connection")
            raise HTTPException(503, "database pool exhausted")
        try:
            import psycopg

            with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=2,
                                 autocommit=True) as conn:
                yield conn
        finally:
            self.slots.release()


db = Database()
