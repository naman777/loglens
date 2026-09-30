"""Window datasets: stream arrays (data/win) + line embeddings (data/emb) -> padded torch batches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

SPLITS = {"train": 0, "val": 1, "test": 2}


class SystemData:
    def __init__(self, system: str, win_dir: str = "data/win", emb_dir: str = "data/emb"):
        self.system = system
        z = np.load(Path(win_dir) / f"{system}.npz", allow_pickle=False)
        self.z = {k: z[k] for k in z.files}
        f = Path(emb_dir) / f"{system}.npy"
        self.emb = np.asarray(np.load(f)) if f.exists() else None  # baselines need no embeddings
        self.n_windows = len(self.z["starts"])
        self.incidents = None
        if "inc_meta" in self.z:
            self.incidents = [tuple(json.loads(s)) for s in self.z["inc_meta"]]  # id, type, tgt, split

    def use_local_templates(self, masked_dir: str = "data/masked", n: int = 2000) -> None:
        """Replace template ids by this system's own top-``n`` templates (ranked by train count) so
        the template-prediction objective can be adapted to a system unseen during pretraining
        without any labels. Everything outside the top-n maps to the 'other' id ``n``."""
        import polars as pl

        u = pl.read_parquet(Path(masked_dir) / self.system / "uniques.parquet",
                            columns=["uid", "count_train"]).sort("uid")
        order = np.argsort(-u["count_train"].to_numpy(), kind="stable")[:n]
        lut = np.full(len(u), n, dtype=np.int64)
        lut[order] = np.arange(len(order))
        self.z["tid"] = lut[self.z["uid"]]

    def windows(self, split: str, labeled: str = "any") -> np.ndarray:
        """Window indices of a split. labeled: any | normal | anomalous."""
        m = self.z["wsplit"] == SPLITS[split]
        if labeled == "normal":
            m &= self.z["wlabel"] == 0
        elif labeled == "anomalous":
            m &= self.z["wlabel"] == 1
        return np.nonzero(m)[0]

    def span(self, w: int) -> tuple[int, int]:
        s = int(self.z["starts"][w])
        return s, s + int(self.z["lens"][w])

    def slice_batch(self, spans: list[tuple[int, int]]) -> dict[str, np.ndarray]:
        L = max(b - a for a, b in spans)
        B = len(spans)
        d = self.emb.shape[1] if self.emb is not None else 256
        emb = np.zeros((B, L, d), dtype=np.float16)
        gap = np.zeros((B, L), dtype=np.int64)
        svc = np.zeros((B, L), dtype=np.int64)
        lvl = np.zeros((B, L), dtype=np.int64)
        tid = np.zeros((B, L), dtype=np.int64)
        causal = np.zeros((B, L), dtype=np.float32)
        pad = np.ones((B, L), dtype=bool)
        for i, (a, b) in enumerate(spans):
            n = b - a
            emb[i, :n] = self.emb[self.z["uid"][a:b]]
            gap[i, :n] = self.z["gap"][a:b]
            svc[i, :n] = self.z["svc"][a:b]
            lvl[i, :n] = self.z["lvl"][a:b]
            tid[i, :n] = self.z["tid"][a:b]
            causal[i, :n] = self.z["causal"][a:b]
            pad[i, :n] = False
        return {"emb": emb, "gap": gap, "svc": svc, "lvl": lvl, "tid": tid, "causal": causal,
                "pad": pad}

    def batch(self, widx: np.ndarray) -> dict[str, np.ndarray]:
        out = self.slice_batch([self.span(int(w)) for w in widx])
        out["y"] = self.z["wlabel"][widx].astype(np.float32)
        return out


def to_torch(b: dict[str, np.ndarray], dev: str) -> dict[str, torch.Tensor]:
    t = {}
    for k, v in b.items():
        x = torch.from_numpy(v)
        if k == "emb":
            x = x.float()
        t[k] = x.to(dev)
    return t


def load_systems(names: list[str], win_dir: str = "data/win", emb_dir: str = "data/emb",
                 local_templates: bool = False, masked_dir: str = "data/masked"):
    out = {n: SystemData(n, win_dir, emb_dir) for n in names}
    if local_templates:
        for sd in out.values():
            sd.use_local_templates(masked_dir)
    return out


class MixedSampler:
    """Samples (system, window) pairs; system prob ∝ n^power (flattens huge systems)."""

    def __init__(self, systems: dict[str, SystemData], split: str, labeled: str, power: float,
                 rng: np.random.Generator, min_windows: int = 1):
        self.rng = rng
        self.items = {n: s.windows(split, labeled) for n, s in systems.items()}
        self.items = {n: w for n, w in self.items.items() if len(w) >= min_windows}
        self.names = list(self.items)
        p = np.array([len(self.items[n]) ** power for n in self.names], dtype=np.float64)
        self.p = p / p.sum()
        self.systems = systems

    def sample(self, batch_size: int):
        """One batch from a single system (keeps padding tight and tensors homogeneous)."""
        n = self.names[int(self.rng.choice(len(self.names), p=self.p))]
        w = self.rng.choice(self.items[n], size=min(batch_size, len(self.items[n])), replace=False)
        return n, self.systems[n].batch(w)
