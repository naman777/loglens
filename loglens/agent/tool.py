"""On-call agent integration: one tool, ``rank_suspect_lines``.

The agent calls it first on every alert and then reasons only over the returned lines. The tool
definition below is in the Anthropic tool-use schema (also valid as an OpenAI function schema body).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loglens.serve.runtime import LogLensRuntime, RuntimeConfig

TOOL_SPEC = {
    "name": "rank_suspect_lines",
    "description": (
        "Score a service's recent logs for an incident and return the log lines most likely to "
        "explain it. Runs a small local model on CPU; call this FIRST for every alert, then reason "
        "only over the returned lines instead of reading raw logs."),
    "input_schema": {
        "type": "object",
        "properties": {
            "service": {"type": "string", "description": "service name, or 'all' for every service"},
            "window": {"type": "string", "description": "ISO start/end joined by '/', e.g. "
                       "'2026-03-01T10:00:00/2026-03-01T10:05:00'; omit for the latest window"},
            "top_k": {"type": "integer", "default": 15},
        },
        "required": ["service"],
    },
}


@dataclass
class LogSource:
    """Directory of ``*.jsonl`` logs (one object per line with ts/service/level/message)."""
    path: str

    def read(self) -> list[dict]:
        rows = []
        for f in sorted(Path(self.path).rglob("*.jsonl")):
            if f.name.startswith("incidents"):
                continue
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: r.get("ts", 0))
        return rows


class LogLensTool:
    def __init__(self, source: LogSource, runtime: LogLensRuntime | None = None,
                 cfg: RuntimeConfig | None = None):
        self.source = source
        self.rt = runtime or LogLensRuntime(cfg or RuntimeConfig())
        self._rows: list[dict] | None = None

    def rows(self) -> list[dict]:
        if self._rows is None:
            self._rows = self.source.read()
        return self._rows

    def rank_suspect_lines(self, service: str, window: str = "", top_k: int = 15) -> dict:
        from loglens.serve.runtime import _ISO, _iso_ms

        rows = self.rows()
        if window:
            a, _, b = window.partition("/")
            ma, mb = _ISO.search(a), _ISO.search(b)
            lo = _iso_ms(ma) if ma else None
            hi = _iso_ms(mb) if mb else None
            rows = [r for r in rows if (lo is None or r["ts"] >= lo) and (hi is None or r["ts"] <= hi)]
        else:
            rows = rows[-2000:]
        if not rows:
            return {"anomaly_score": 0.0, "n_lines": 0, "top_lines": []}
        res = self.rt.score([json.dumps(r) for r in rows], top_k=top_k if service == "all" else 10 * top_k)
        out = []
        for s in res.lines:
            r = rows[s["index"]]
            if service != "all" and r.get("service") != service:
                continue
            out.append({"ts": r["ts"], "service": r.get("service"), "level": r.get("level"),
                        "score": round(s["score"], 3), "text": r.get("message", "")[:240]})
            if len(out) >= top_k:
                break
        return {"anomaly_score": round(res.anomaly, 4), "n_lines": len(rows), "top_lines": out}
