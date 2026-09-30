import uuid

import httpx
from common import log
from fastapi import FastAPI, HTTPException

app = FastAPI()
UP = {"orders": "http://orders:8000", "payments": "http://payments:8000",
      "inventory": "http://inventory:8000"}


async def call(svc: str, method: str, path: str, rid: str):
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            response = await c.request(method, UP[svc] + path)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        log("ERROR", f"upstream {svc} returned status={exc.response.status_code}", rid=rid)
        raise HTTPException(502, f"{svc} failed") from exc
    except httpx.ConnectError:
        log("ERROR", f"connection refused calling {svc}:8000 for {method} {path}", rid=rid)
        raise HTTPException(503)
    except httpx.TimeoutException:
        log("ERROR", f"timeout after 2000ms calling {svc} for {method} {path}", rid=rid)
        raise HTTPException(504)


@app.get("/products")
async def products():
    rid = uuid.uuid4().hex[:8]
    log("INFO", f"request received method=GET path=/products rid={rid}", rid=rid)
    r = await call("inventory", "GET", "/stock", rid)
    log("INFO", f"request completed status=200 rid={rid}", rid=rid)
    return r


@app.post("/orders")
async def orders():
    rid = uuid.uuid4().hex[:8]
    log("INFO", f"request received method=POST path=/orders rid={rid}", rid=rid)
    return await call("orders", "POST", "/create", rid)


@app.post("/pay")
async def pay():
    rid = uuid.uuid4().hex[:8]
    log("INFO", f"request received method=POST path=/pay rid={rid}", rid=rid)
    return await call("payments", "POST", "/charge", rid)


@app.get("/health")
def health():
    return {"ok": True}
