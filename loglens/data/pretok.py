"""Pre-tokenize each system's unique masked lines to flat int16 shards (ragged: ids + offsets).

data/tok/<system>.npz: uid (n,), offsets (n+1,), ids (total,), count_train (n,)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from loglens.config import load_config, parse_args
from loglens.tokenizer.tok import LogTokenizer
from loglens.tokenizer.train_bpe import prefixed


@dataclass
class PretokConfig:
    masked_dir: str = "data/masked"
    tok_dir: str = "data/tok"
    tokenizer: str = "artifacts/tokenizer/loglens-bpe-16k.json"
    max_len: int = 128
    systems: list[str] = field(default_factory=list)
    chunk: int = 200_000


def tokenize_system(system: str, cfg: PretokConfig, tk: LogTokenizer) -> dict:
    u = pl.read_parquet(Path(cfg.masked_dir) / system / "uniques.parquet")
    texts = [prefixed(lv, sv, m) for lv, sv, m in zip(u["level"], u["svc"], u["masked"], strict=True)]
    flat: list[np.ndarray] = []
    lens: list[int] = []
    for i in range(0, len(texts), cfg.chunk):
        for ids in tk.encode_ids(texts[i: i + cfg.chunk]):
            flat.append(np.asarray(ids, dtype=np.int16))
            lens.append(len(ids))
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    out = Path(cfg.tok_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{system}.npz", uid=u["uid"].to_numpy(), offsets=offsets,
             ids=np.concatenate(flat) if flat else np.zeros(0, np.int16),
             count_train=u["count_train"].to_numpy())
    la = np.asarray(lens)
    return {"unique": len(lens), "mean_len": float(la.mean()), "tokens": int(la.sum())}


def main(cfg: PretokConfig) -> None:
    tk = LogTokenizer(cfg.tokenizer, cfg.max_len)
    root = Path(cfg.masked_dir)
    systems = cfg.systems or sorted(p.name for p in root.iterdir() if p.is_dir())
    for s in systems:
        print(s, tokenize_system(s, cfg, tk), flush=True)


if __name__ == "__main__":
    args = parse_args("pre-tokenize uniques")
    main(load_config(PretokConfig, args.config))
