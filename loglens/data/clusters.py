"""Drain3 cluster id per distinct masked line -> data/clusters/<system>.npy (row == uid)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from loglens.eval.baselines.classic import drain_clusters


def main(masked_dir: str = "data/masked", out_dir: str = "data/clusters") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(masked_dir).iterdir()):
        if p.is_dir():
            c = drain_clusters(p.name, masked_dir, 400_000)
            np.save(Path(out_dir) / f"{p.name}.npy", c.astype(np.int32))
            print(p.name, len(c), len(set(c.tolist())), flush=True)


if __name__ == "__main__":
    main()
