"""Embed every distinct masked line once with the (frozen) line encoder -> data/emb/<system>.npy.

Row index == uid. float16, (n_unique, embed_dim).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from loglens.config import load_config, parse_args
from loglens.model.line_encoder import LineEncoder, LineEncoderConfig
from loglens.seed import get_device


@dataclass
class EmbedConfig:
    encoder: str = "artifacts/pretrain/encoder.pt"
    tok_dir: str = "data/tok"
    emb_dir: str = "data/emb"
    batch_tokens: int = 40_000
    systems: list[str] = field(default_factory=list)


def load_encoder(path: str, dev: str = "cpu") -> LineEncoder:
    if path == "random":  # ablation: untrained encoder with the same architecture
        torch.manual_seed(1337)
        return LineEncoder(LineEncoderConfig()).to(dev).eval()
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = LineEncoderConfig(**ck["cfg"])
    m = LineEncoder(cfg)
    m.load_state_dict(ck["model"])
    return m.to(dev).eval()


@torch.no_grad()
def embed_ragged(model: LineEncoder, ids: np.ndarray, offsets: np.ndarray, dev: str,
                 batch_tokens: int = 40_000, max_len: int | None = None) -> np.ndarray:
    n = len(offsets) - 1
    lens = np.minimum(np.diff(offsets), max_len or model.cfg.max_len)
    order = np.argsort(lens, kind="stable")
    out = np.zeros((n, model.cfg.embed_dim), dtype=np.float16)
    i = 0
    while i < n:
        j = i
        while j < n and (j - i + 1) * lens[order[j]] <= batch_tokens:
            j += 1
        j = max(j, i + 1)
        rows = order[i:j]
        width = int(lens[rows].max())
        batch = np.zeros((len(rows), width), dtype=np.int64)
        for r, k in enumerate(rows):
            ln = lens[k]
            batch[r, :ln] = ids[offsets[k]: offsets[k] + ln]
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            e = model(torch.from_numpy(batch).to(dev))
        out[rows] = e.float().cpu().numpy().astype(np.float16)
        i = j
    return out


def main(cfg: EmbedConfig) -> None:
    dev = get_device()
    model = load_encoder(cfg.encoder, dev)
    Path(cfg.emb_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(cfg.tok_dir).glob("*.npz"))
    for f in files:
        if cfg.systems and f.stem not in cfg.systems:
            continue
        z = np.load(f)
        emb = embed_ragged(model, z["ids"], z["offsets"], dev, cfg.batch_tokens)
        np.save(Path(cfg.emb_dir) / f"{f.stem}.npy", emb)
        print(f.stem, emb.shape, flush=True)


if __name__ == "__main__":
    args = parse_args("embed uniques")
    main(load_config(EmbedConfig, args.config))
