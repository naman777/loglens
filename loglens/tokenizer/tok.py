"""Thin wrapper around a trained log BPE: fixed-length padded id batches."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from loglens.tokenizer.masking import DURATION_BUCKETS, LEVEL_TOKENS, MASK_TOKENS

SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
DEFAULT_SERVICES = [
    "BGL", "HDFS", "Thunderbird", "OpenStack", "Hadoop", "Spark", "Zookeeper", "Linux", "Apache",
    "HPC", "SSH", "Mac", "HealthApp", "Proxifier", "Android", "Lab", "gateway", "orders", "payments", "inventory", "worker", "postgres", "redis",
    "other",
]


def structural_tokens(services: list[str] | None = None) -> list[str]:
    svc = [f"<SVC:{s}>" for s in (services or DEFAULT_SERVICES)]
    return [*MASK_TOKENS, *LEVEL_TOKENS, *svc]


class LogTokenizer:
    PAD, CLS, SEP, MASK, UNK = range(5)

    def __init__(self, path: str | Path, max_len: int = 128):
        self.tk = Tokenizer.from_file(str(path))
        self.max_len = max_len
        self.tk.no_padding()
        self.tk.no_truncation()
        assert self.tk.token_to_id("[PAD]") == self.PAD

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    def encode_ids(self, texts: list[str]) -> list[list[int]]:
        """Unpadded ids, truncated to ``max_len`` keeping [CLS] first and [SEP] last."""
        out = []
        for e in self.tk.encode_batch(texts):
            ids = e.ids
            if len(ids) > self.max_len:
                ids = ids[: self.max_len - 1] + [self.SEP]
            out.append(ids)
        return out

    def encode_padded(self, texts: list[str]) -> np.ndarray:
        ids = self.encode_ids(texts)
        arr = np.zeros((len(ids), self.max_len), dtype=np.int16)
        for i, row in enumerate(ids):
            arr[i, : len(row)] = row
        return arr


__all__ = ["LogTokenizer", "SPECIALS", "DEFAULT_SERVICES", "structural_tokens", "DURATION_BUCKETS"]
