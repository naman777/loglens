"""Experiment tracking: W&B when available, otherwise a local JSONL log.

Every run records config, git commit hash and (optionally) the dataset manifest hash.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def file_hash(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


class Tracker:
    def __init__(self, project: str, name: str, config: Any, manifest: str | Path | None = None,
                 out_dir: str | Path = "results/runs", use_wandb: bool = False):
        cfg = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config or {})
        cfg["git_commit"] = git_commit()
        cfg["manifest_hash"] = file_hash(manifest) if manifest else None
        self.config = cfg
        self.path = Path(out_dir) / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wandb = None
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb.init(project=project, name=name, config=cfg,
                                         tags=[cfg["git_commit"]])
            except Exception:
                self._wandb = None
        self._write({"_type": "config", **cfg})

    def _write(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), **rec}) + "\n")

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._write({"_type": "metrics", "step": step, **metrics})
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
