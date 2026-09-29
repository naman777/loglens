"""Classic baselines on the shared window splits: Drain3 templates + DeepLog, isolation forest on
template counts, and simple RCA heuristics. All see exactly the same windows as LogLens."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from loglens.train.windata import SystemData


def drain_clusters(system: str, masked_dir: str = "data/masked", max_unique: int = 400_000) -> np.ndarray:
    """Drain3 cluster id for every unique masked line (row index == uid)."""
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig

    logging.getLogger("drain3").setLevel(logging.ERROR)
    u = pl.read_parquet(Path(masked_dir) / system / "uniques.parquet").sort("uid")
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    cfg.drain_sim_th = 0.5
    cfg.drain_depth = 4
    tm = TemplateMiner(config=cfg)
    ids = np.zeros(len(u), dtype=np.int32)
    for i, m in enumerate(u["masked"][:max_unique]):
        ids[i] = tm.add_log_message(m)["cluster_id"]
    for i in range(max_unique, len(u)):
        c = tm.match(u["masked"][i])
        ids[i] = c.cluster_id if c is not None else 0
    return ids


class TemplateSeq:
    """A system's template-id stream (Drain3 cluster per line) with a compact vocabulary."""

    def __init__(self, sd: SystemData, clusters: np.ndarray, train_split: int = 0):
        self.sd = sd
        raw = clusters[sd.z["uid"]]
        tr = sd.z["split"] == train_split
        vocab = np.unique(raw[tr])
        self.vocab = {int(c): i + 1 for i, c in enumerate(vocab)}  # 0 = unseen
        lut = np.zeros(int(raw.max()) + 2, dtype=np.int64)
        for c, i in self.vocab.items():
            lut[c] = i
        self.seq = lut[raw]
        self.V = len(vocab) + 1


class DeepLog(nn.Module):
    def __init__(self, V: int, emb: int = 32, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, V)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.out(h[:, -1])


def _contexts(seq: np.ndarray, a: int, b: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    s = seq[a:b]
    if len(s) <= h:
        return np.zeros((0, h), np.int64), np.zeros(0, np.int64)
    idx = np.arange(h)[None, :] + np.arange(len(s) - h)[:, None]
    return s[idx], s[h:]


class DeepLogBaseline:
    name = "Drain3+DeepLog"

    def __init__(self, sd: SystemData, clusters: np.ndarray, h: int = 10, topk: int = 9,
                 epochs: int = 3, max_samples: int = 400_000, dev: str | None = None, seed: int = 0):
        self.h, self.topk, self.epochs, self.max_samples, self.seed = h, topk, epochs, max_samples, seed
        self.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ts = TemplateSeq(sd, clusters)
        self.sd = sd

    def fit(self) -> "DeepLogBaseline":
        sd, ts, h = self.sd, self.ts, self.h
        X, Y = [], []
        for w in sd.windows("train", "normal"):
            a, b = sd.span(int(w))
            x, y = _contexts(ts.seq, a, b, h)
            X.append(x)
            Y.append(y)
        X, Y = np.concatenate(X), np.concatenate(Y)
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_samples:
            keep = rng.choice(len(X), self.max_samples, replace=False)
            X, Y = X[keep], Y[keep]
        torch.manual_seed(self.seed)
        self.model = DeepLog(ts.V).to(self.dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 1024):
                j = perm[i: i + 1024]
                loss = nn.functional.cross_entropy(self.model(Xt[j].to(self.dev)), Yt[j].to(self.dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def _line_surprise(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-line (miss flag, NLL); the first h lines get zero."""
        self.model.eval()
        x, y = _contexts(self.ts.seq, a, b, self.h)
        miss = np.zeros(b - a)
        nll = np.zeros(b - a)
        if len(x) == 0:
            return miss, nll
        lg = self.model(torch.from_numpy(x).to(self.dev)).float()
        yt = torch.from_numpy(y).to(self.dev)
        top = lg.topk(min(self.topk, lg.size(1)), dim=1).indices
        m = ~(top == yt[:, None]).any(1)
        miss[self.h:] = m.cpu().numpy()
        nll[self.h:] = nn.functional.cross_entropy(lg, yt, reduction="none").cpu().numpy()
        return miss, nll

    def score_windows(self, widx: np.ndarray) -> np.ndarray:
        out = np.zeros(len(widx))
        for k, w in enumerate(widx):
            a, b = self.sd.span(int(w))
            miss, nll = self._line_surprise(a, b)
            n = max(b - a - self.h, 1)
            out[k] = miss.sum() / n + 1e-4 * nll.max()
        return out

    def rank_lines(self, a: int, b: int) -> np.ndarray:
        miss, nll = self._line_surprise(a, b)
        return miss * 1000 + nll


class TemplateCountIForest:
    name = "Drain3+IsolationForest"

    def __init__(self, sd: SystemData, clusters: np.ndarray, max_features: int = 512, seed: int = 0):
        self.sd, self.seed = sd, seed
        self.ts = TemplateSeq(sd, clusters)
        self.F = min(self.ts.V, max_features)

    def _feat(self, widx: np.ndarray) -> np.ndarray:
        X = np.zeros((len(widx), self.F), dtype=np.float32)
        for k, w in enumerate(widx):
            a, b = self.sd.span(int(w))
            s = np.minimum(self.ts.seq[a:b], self.F - 1)
            X[k] = np.bincount(s, minlength=self.F)
        return np.log1p(X)

    def fit(self):
        from sklearn.ensemble import IsolationForest

        w = self.sd.windows("train", "normal")
        if len(w) > 20000:
            w = np.random.default_rng(self.seed).choice(w, 20000, replace=False)
        self.model = IsolationForest(n_estimators=100, random_state=self.seed, n_jobs=-1).fit(self._feat(w))
        return self

    def score_windows(self, widx: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(self._feat(widx))


class TemplateCountLogReg(TemplateCountIForest):
    """Supervised baseline: logistic regression on the same log-count vectors, trained on the
    system's labelled train windows (the fair comparison for LogLens-sup)."""

    name = "Drain3+LogReg (supervised)"

    def fit(self):
        from sklearn.linear_model import LogisticRegression

        w = self.sd.windows("train")
        if len(w) > 60000:
            w = np.random.default_rng(self.seed).choice(w, 60000, replace=False)
        y = self.sd.z["wlabel"][w]
        self.model = LogisticRegression(max_iter=300, class_weight="balanced", C=1.0).fit(self._feat(w), y)
        return self

    def score_windows(self, widx: np.ndarray) -> np.ndarray:
        return self.model.decision_function(self._feat(widx))


_SEV = {"FATAL": 5, "CRITICAL": 5, "ERROR": 4, "WARN": 3, "NOTICE": 2, "INFO": 1, "DEBUG": 0,
        "TRACE": 0, "UNK": 1}


def severity_rank(sd: SystemData, a: int, b: int, levels: list[str]) -> np.ndarray:
    """Heuristic RCA baseline: severity, ties broken by template rarity in the slice."""
    lv = np.array([_SEV.get(levels[i], 1) for i in sd.z["lvl"][a:b]], dtype=np.float64)
    tid = sd.z["tid"][a:b]
    _, inv, cnt = np.unique(tid, return_inverse=True, return_counts=True)
    return lv + 0.5 / cnt[inv]


def random_rank(a: int, b: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(b - a)
