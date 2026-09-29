"""HTTP API: POST /score, GET /health, GET /metrics."""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from loglens.serve.runtime import LogLensRuntime, RuntimeConfig

app = FastAPI(title="LogLens", version="0.1.0")
_rt: LogLensRuntime | None = None


def runtime() -> LogLensRuntime:
    global _rt
    if _rt is None:
        _rt = LogLensRuntime(RuntimeConfig(
            model_dir=os.environ.get("LOGLENS_MODEL_DIR", "artifacts/onnx"),
            tokenizer=os.environ.get("LOGLENS_TOKENIZER", "artifacts/tokenizer/loglens-bpe-16k.json"),
            int8=os.environ.get("LOGLENS_INT8", "1") == "1",
            threads=int(os.environ.get("LOGLENS_THREADS", "4"))))
    return _rt


class ScoreRequest(BaseModel):
    lines: list[str]
    top_k: int = 15


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    r = runtime().score(req.lines, req.top_k)
    return {"anomaly_score": r.anomaly, "n_lines": r.n_lines, "window_scores": r.window_scores,
            "suspects": r.lines}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    return runtime().metrics()
