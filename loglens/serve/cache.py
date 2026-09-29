"""LRU embedding cache keyed by a hash of the masked line text (float16, 256-d ≈ 0.5 KB/entry)."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


class EmbeddingCache:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self._d: OrderedDict[int, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: int) -> np.ndarray | None:
        v = self._d.get(key)
        if v is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key: int, value: np.ndarray) -> None:
        self._d[key] = value.astype(np.float16, copy=False)
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def reset_stats(self) -> None:
        self.hits = self.misses = 0
