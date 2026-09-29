"""YAML config loading into dataclasses. Every entry point takes ``--config``."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_config(cls: type[T], path: str | Path | None) -> T:
    """Load ``path`` into dataclass ``cls``; unknown keys raise, missing keys use defaults."""
    data: dict[str, Any] = {}
    if path:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def parse_args(description: str = "") -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to a YAML config")
    return p.parse_args()
