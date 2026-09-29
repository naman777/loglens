"""CPU inference runtime: parse -> mask -> cache lookup -> batch misses through the encoder ->
window model. Only needs onnxruntime + tokenizers + numpy (no PyTorch)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from loglens.serve.cache import EmbeddingCache
from loglens.tokenizer.masking import LEVELS, level_token, mask, service_token
from loglens.tokenizer.tok import DEFAULT_SERVICES

_LEVEL_RE = re.compile(r"\b(TRACE|DEBUG|INFO|NOTICE|WARN(?:ING)?|ERROR|ERR|FATAL|CRITICAL|SEVERE)\b")
_ISO = re.compile(r"(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)(?:[.,](\d+))?")
_SVC = re.compile(
    r"\[([A-Za-z][\w.\-]{1,30})\]|^\w{3} +\d+ \d\d:\d\d:\d\d \S+ ([^\s:\[]+)(?:\[\d+\])?:"
    r"|^\S+ \S+ (\S+?)(?:\[\d+\])?: ")


@dataclass
class Parsed:
    ts: int | None
    level: str
    service: str
    message: str


def parse_line(line: str) -> Parsed:
    s = line.strip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
            msg = str(d.get("message") or d.get("msg") or d.get("log") or s)
            lv = str(d.get("level") or d.get("severity") or "").upper()
            svc = str(d.get("service") or d.get("component") or d.get("logger") or "other")
            ts = d.get("ts") or d.get("timestamp") or d.get("time")
            if isinstance(ts, str):
                m = _ISO.search(ts)
                ts = _iso_ms(m) if m else None
            elif isinstance(ts, (int, float)):
                ts = int(ts if ts > 1e11 else ts * 1000)
            return Parsed(ts, lv or "UNK", svc, msg)
        except json.JSONDecodeError:
            pass
    m = _ISO.search(s[:40])
    ts = _iso_ms(m) if m else None
    lm = _LEVEL_RE.search(s[:120])
    lv = lm.group(1) if lm else "UNK"
    sm = _SVC.search(s[:160])
    svc = (sm.group(1) or sm.group(2) or sm.group(3)) if sm else "other"
    return Parsed(ts, lv, svc, s)


def _iso_ms(m: re.Match) -> int:
    from datetime import datetime, timezone

    y, mo, d, h, mi, se, frac = m.groups()
    dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(se), tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + (int(frac[:3].ljust(3, "0")) if frac else 0)


def _mask_chunk(msgs: list[str]) -> list[str]:
    return [mask(m) for m in msgs]


@dataclass
class RuntimeConfig:
    model_dir: str = "artifacts/onnx"
    tokenizer: str = "artifacts/tokenizer/loglens-bpe-16k.json"
    int8: bool = True
    threads: int = 4
    cache_size: int = 200_000
    micro_batch: int = 512
    window: int = 256
    stride: int = 256
    max_len: int = 128
    services: list[str] = field(default_factory=lambda: list(DEFAULT_SERVICES))
    use_cache: bool = True
    mask_workers: int = 0  # >0: mask uncached messages in a process pool


@dataclass
class Scored:
    anomaly: float
    lines: list[dict]  # top-k suspects: rank, index, score, ts, text
    n_lines: int
    window_scores: list[float]


class LogLensRuntime:
    def __init__(self, cfg: RuntimeConfig):
        import onnxruntime as ort

        self.cfg = cfg
        d = Path(cfg.model_dir)
        suffix = ".int8.onnx" if cfg.int8 else ".onnx"
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.threads
        so.inter_op_num_threads = 1
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(d / f"encoder{suffix}"), so, providers=prov)
        self.win = ort.InferenceSession(str(d / f"window{suffix}"), so, providers=prov)
        self.tk = Tokenizer.from_file(cfg.tokenizer)
        self.tk.no_padding()
        self.tk.no_truncation()
        self.cache = EmbeddingCache(cfg.cache_size)
        self.mask_cache: dict[str, str] = {}
        self.svc_ix = {s: i for i, s in enumerate(cfg.services)}
        self.svc_known = set(cfg.services)
        self.stats = {"lines": 0, "encoder_lines": 0, "secs": 0.0}
        self._pool = None

    def _prefill_masks(self, msgs: list[str]) -> None:
        """Mask all not-yet-cached messages, in parallel when a worker pool is configured."""
        todo = list({m for m in msgs if m not in self.mask_cache})
        if not todo:
            return
        if self.cfg.mask_workers > 0 and len(todo) >= 2000:
            if self._pool is None:
                from concurrent.futures import ProcessPoolExecutor

                self._pool = ProcessPoolExecutor(self.cfg.mask_workers)
            k = self.cfg.mask_workers * 2
            chunks = [todo[i::k] for i in range(k)]
            out = list(self._pool.map(_mask_chunk, chunks))
            for ch, res in zip(chunks, out, strict=True):
                for m, r in zip(ch, res, strict=True):
                    self.mask_cache[m] = r
        else:
            for m in todo:
                self.mask_cache[m] = mask(m)
        if len(self.mask_cache) > 500_000:
            self.mask_cache.clear()

    # ---------------------------------------------------------------- embeddings
    def _text(self, p: Parsed) -> str:
        key = p.message
        m = self.mask_cache.get(key)
        if m is None:
            m = mask(p.message)
        return f"{level_token(p.level)} {service_token(p.service, self.svc_known)} {m}"

    def embed(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        out = np.zeros((n, 256), dtype=np.float32)
        keys = [hash(t) for t in texts]
        todo: dict[int, int] = {}  # key -> first row needing compute
        rows_for: dict[int, list[int]] = {}
        for i, k in enumerate(keys):
            v = self.cache.get(k) if self.cfg.use_cache else None
            if v is not None:
                out[i] = v
            else:
                rows_for.setdefault(k, []).append(i)
                todo.setdefault(k, i)
        if todo:
            uniq = list(todo.items())
            for s in range(0, len(uniq), self.cfg.micro_batch):
                chunk = uniq[s: s + self.cfg.micro_batch]
                enc = self.tk.encode_batch([texts[i] for _, i in chunk])
                L = min(max(len(e.ids) for e in enc), self.cfg.max_len)
                ids = np.zeros((len(chunk), L), dtype=np.int64)
                for r, e in enumerate(enc):
                    x = e.ids[:L] if len(e.ids) <= L else e.ids[: L - 1] + [2]
                    ids[r, : len(x)] = x
                emb = self.enc.run(None, {"ids": ids})[0]
                for (k, _), e in zip(chunk, emb, strict=True):
                    if self.cfg.use_cache:
                        self.cache.put(k, e)
                    for i in rows_for[k]:
                        out[i] = e
            self.stats["encoder_lines"] += len(uniq)
        return out

    # ---------------------------------------------------------------- scoring
    def score(self, raw_lines: list[str], top_k: int = 15) -> Scored:
        t0 = time.perf_counter()
        parsed = [parse_line(x) for x in raw_lines]
        self._prefill_masks([p.message for p in parsed])
        emb = self.embed([self._text(p) for p in parsed])
        n = len(parsed)
        ts = [p.ts for p in parsed]
        gap = np.zeros(n, dtype=np.int64)
        for i in range(1, n):
            if ts[i] is not None and ts[i - 1] is not None:
                g = max(ts[i] - ts[i - 1], 0)
                gap[i] = min(int(np.floor(np.log2(g + 1))), 15)
        svc = np.array([self.svc_ix.get(p.service, self.svc_ix["other"]) for p in parsed], dtype=np.int64)
        lvl = np.array([LEVELS.index(_norm_level(p.level)) for p in parsed], dtype=np.int64)
        W, S = self.cfg.window, self.cfg.stride
        spans, s = [], 0
        while True:
            e = min(s + W, n)
            spans.append((s, e))
            if e >= n:
                break
            s += S
        susp = np.full(n, -1e9)
        wscore = []
        for a, b in spans:
            L = b - a
            logit, sl = self.win.run(None, {
                "emb": emb[None, a:b], "gap": gap[None, a:b], "svc": svc[None, a:b],
                "lvl": lvl[None, a:b], "pad": np.zeros((1, L), dtype=bool)})
            wscore.append(float(1 / (1 + np.exp(-logit[0]))))
            susp[a:b] = np.maximum(susp[a:b], sl[0])
        order = np.argsort(-susp)[:top_k]
        top = [{"rank": r + 1, "index": int(i), "score": float(susp[i]), "ts": ts[i],
                "text": raw_lines[i][:300]} for r, i in enumerate(order)]
        self.stats["lines"] += n
        self.stats["secs"] += time.perf_counter() - t0
        return Scored(max(wscore), top, n, wscore)

    def metrics(self) -> dict:
        s = self.stats
        return {"lines": s["lines"], "lines_per_s": s["lines"] / s["secs"] if s["secs"] else 0.0,
                "cache_hit_rate": self.cache.hit_rate, "cache_entries": len(self.cache),
                "encoder_lines": s["encoder_lines"]}


def _norm_level(lv: str) -> str:
    lv = (lv or "UNK").upper()
    lv = {"WARNING": "WARN", "ERR": "ERROR", "SEVERE": "ERROR"}.get(lv, lv)
    return lv if lv in LEVELS else "UNK"
