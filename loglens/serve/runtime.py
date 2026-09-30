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


_KNOWN = frozenset(DEFAULT_SERVICES)


def _prepare_chunk(lines: list[str], known: frozenset[str] = _KNOWN) -> list[tuple]:
    """parse + mask + prefix for a chunk of raw lines -> (ts, service, level_idx, text)."""
    out = []
    memo: dict[str, str] = {}
    for line in lines:
        p = parse_line(line)
        m = memo.get(p.message)
        if m is None:
            m = memo[p.message] = mask(p.message)
        lv = _norm_level(p.level)
        out.append((p.ts, p.service, LEVELS.index(lv),
                    f"{level_token(lv)} {service_token(p.service, known)} {m}"))
    return out


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
        self.svc_ix = {s: i for i, s in enumerate(cfg.services)}
        self.svc_known = set(cfg.services)
        self.stats = {"lines": 0, "encoder_lines": 0, "secs": 0.0}
        self._pool = None

    def _submit(self, raw_lines: list[str]):
        """Start parse+mask for a batch; returns a handle for ``_gather`` (pool futures, or rows)."""
        w = self.cfg.mask_workers
        if w > 0 and len(raw_lines) >= 4000:
            if self._pool is None:
                from concurrent.futures import ProcessPoolExecutor

                self._pool = ProcessPoolExecutor(w)
            n = len(raw_lines)
            k = w * 2
            return [self._pool.submit(_prepare_chunk, raw_lines[n * i // k: n * (i + 1) // k], self.svc_known)
                    for i in range(k)]
        return _prepare_chunk(raw_lines, self.svc_known)

    @staticmethod
    def _gather(handle) -> list[tuple]:
        if isinstance(handle, list) and handle and hasattr(handle[0], "result"):
            return [row for f in handle for row in f.result()]
        return handle

    def _prepare(self, raw_lines: list[str]) -> list[tuple]:
        return self._gather(self._submit(raw_lines))

    # ---------------------------------------------------------------- embeddings
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
        return self._score_rows(raw_lines, self._prepare(raw_lines), top_k, t0)

    def score_stream(self, batches, top_k: int = 15):
        """Yield a Scored per batch, parsing/masking batch i+1 in the pool while scoring batch i."""
        it = iter(batches)
        nxt = next(it, None)
        handle = self._submit(nxt) if nxt is not None else None
        while nxt is not None:
            cur, cur_handle = nxt, handle
            nxt = next(it, None)
            handle = self._submit(nxt) if nxt is not None else None
            t0 = time.perf_counter()
            yield self._score_rows(cur, self._gather(cur_handle), top_k, t0)

    def _score_rows(self, raw_lines: list[str], rows: list[tuple], top_k: int, t0: float) -> Scored:
        emb = self.embed([r[3] for r in rows])
        n = len(rows)
        ts = [r[0] for r in rows]
        tsa = np.array([t if t is not None else np.nan for t in ts], dtype=np.float64)
        d = np.zeros(n)
        if n > 1:
            d[1:] = np.maximum(np.diff(tsa), 0)
        gap = np.minimum(np.floor(np.log2(np.nan_to_num(d, nan=0.0) + 1)), 15).astype(np.int64)
        other = self.svc_ix["other"]
        svc = np.array([self.svc_ix.get(r[1], other) for r in rows], dtype=np.int64)
        lvl = np.array([r[2] for r in rows], dtype=np.int64)
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
