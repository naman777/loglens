"""Build ordered line streams and windows for the window model.

Session systems (HDFS): one window per block session (first ``window`` lines).
Everything else: sliding count windows (``window`` lines, stride ``stride``) inside each split.
Window label = any line labelled anomalous (HDFS: the block label).
Lab additionally carries per-line causal labels and an incident table (slice into the stream).

Output ``data/win/<system>.npz``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from loglens.config import load_config, parse_args
from loglens.tokenizer.masking import LEVELS
from loglens.tokenizer.tok import DEFAULT_SERVICES

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
SESSION_SYSTEMS = {"HDFS"}
N_TEMPLATES = 2000


@dataclass
class WindowsConfig:
    masked_dir: str = "data/masked"
    lab_dir: str = "data/lab"
    win_dir: str = "data/win"
    window: int = 256
    stride: int = 128
    unseen: str = "Thunderbird"
    systems: list[str] = field(default_factory=list)
    services: list[str] = field(default_factory=lambda: list(DEFAULT_SERVICES))
    n_templates: int = N_TEMPLATES
    # per-system window length (stride = length): Thunderbird alerts are ~5-8% of lines, so 256-line
    # windows are 96% positive in val/test and F1 becomes meaningless; 20-line windows are the
    # convention in the BGL/Thunderbird literature.
    window_override: dict[str, int] = field(default_factory=lambda: {"Thunderbird": 20})
    incident_pre_s: int = 30
    incident_post_s: int = 60


def service_index(services: list[str]) -> dict[str, int]:
    return {s: i for i, s in enumerate(services)}


def gap_bucket(gap_ms: np.ndarray) -> np.ndarray:
    """log2-bucketed gap since previous line: 0 ms -> 0 ... >= 2^14 ms -> 15."""
    return np.clip(np.floor(np.log2(np.maximum(gap_ms, 0) + 1)), 0, 15).astype(np.int8)


def build_template_table(cfg: WindowsConfig) -> dict[tuple[str, str, str], int]:
    """Top-N (svc, level, masked) by train count over non-unseen systems -> template id."""
    parts = []
    for p in Path(cfg.masked_dir).iterdir():
        if p.is_dir() and p.name != cfg.unseen:
            parts.append(pl.read_parquet(p / "uniques.parquet", columns=["svc", "level", "masked",
                                                                         "count_train"]))
    top = pl.concat(parts).group_by(["svc", "level", "masked"]).agg(
        pl.col("count_train").sum().alias("c")).sort("c", descending=True).head(cfg.n_templates)
    return {(a, b, c): i for i, (a, b, c) in enumerate(zip(top["svc"], top["level"], top["masked"],
                                                            strict=True))}


def sliding(starts_ends: list[tuple[int, int]], window: int, stride: int) -> list[tuple[int, int]]:
    out = []
    for a, b in starts_ends:
        if b - a <= 0:
            continue
        s = a
        while True:
            e = min(s + window, b)
            out.append((s, e))
            if e >= b:
                break
            s += stride
    return out


def build_system(system: str, cfg: WindowsConfig, templates: dict) -> dict:
    root = Path(cfg.masked_dir) / system
    lines = pl.read_parquet(root / "lines.parquet")
    uniq = pl.read_parquet(root / "uniques.parquet")
    svc_ix = service_index(cfg.services)
    other = svc_ix.get("other", len(cfg.services) - 1)
    tid = np.array([templates.get((a, b, c), cfg.n_templates)
                    for a, b, c in zip(uniq["svc"], uniq["level"], uniq["masked"], strict=True)],
                   dtype=np.int16)
    u_svc = np.array([svc_ix.get(s, other) for s in uniq["svc"]], dtype=np.int16)
    u_lvl = np.array([LEVELS.index(lv) if lv in LEVELS else LEVELS.index("UNK")
                      for lv in uniq["level"]], dtype=np.int8)
    if system == "Lab":
        lab = pl.read_parquet(Path(cfg.lab_dir) / "lines.parquet",
                              columns=["line_no", "is_causal", "incident_id"])
        lines = lines.join(lab, on="line_no", how="left")
    else:
        lines = lines.with_columns(pl.lit(False).alias("is_causal"),
                                   pl.lit(None, dtype=pl.Utf8).alias("incident_id"))
    lines = lines.with_columns(pl.col("split").replace_strict(SPLIT_CODE).cast(pl.Int8).alias("sp"))
    if system in SESSION_SYSTEMS:
        lines = lines.filter(pl.col("session_id").is_not_null())
        lines = lines.with_columns(pl.col("ts").min().over("session_id").alias("s0"))
        lines = lines.sort(["s0", "session_id", "ts", "line_no"])
    else:
        lines = lines.sort(["ts", "line_no"])
    ts = lines["ts"].to_numpy()
    uid = lines["uid"].to_numpy().astype(np.int32)
    sp = lines["sp"].to_numpy()
    lab_line = lines["label"].fill_null(0).to_numpy().astype(np.int8)
    causal = lines["is_causal"].fill_null(False).to_numpy().astype(np.int8)
    gap = np.diff(ts, prepend=ts[:1])
    if system in SESSION_SYSTEMS:
        sess = lines["session_id"].to_numpy()
        first = np.concatenate([[True], sess[1:] != sess[:-1]])
        gap = np.where(first, 0, gap)
        idx = np.nonzero(first)[0]
        bounds = list(zip(idx, np.concatenate([idx[1:], [len(sess)]]), strict=True))
        wins = [(a, min(b, a + cfg.window)) for a, b in bounds]
    else:
        # contiguous segments per split (stream is time-ordered, splits are time ranges)
        cuts = np.nonzero(np.diff(sp) != 0)[0] + 1
        seg = list(zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(sp)]]), strict=True))
        ov = cfg.window_override.get(system)
        wins = sliding(seg, ov, ov) if ov else sliding(seg, cfg.window, cfg.stride)
    starts = np.array([w[0] for w in wins], dtype=np.int64)
    lens = np.array([w[1] - w[0] for w in wins], dtype=np.int32)
    csum = np.concatenate([[0], np.cumsum(lab_line, dtype=np.int64)])
    wlabel = ((csum[starts + lens] - csum[starts]) > 0).astype(np.int8)
    ccs = np.concatenate([[0], np.cumsum(causal, dtype=np.int64)])
    wcausal = ((ccs[starts + lens] - ccs[starts]) > 0).astype(np.int8)
    wsplit = sp[starts]
    arrays = dict(
        uid=uid, gap=gap_bucket(gap), svc=u_svc[uid], lvl=u_lvl[uid], tid=tid[uid], label=lab_line,
        causal=causal, ts=ts, split=sp, starts=starts, lens=lens, wlabel=wlabel, wcausal=wcausal,
        wsplit=wsplit,
    )
    inc_rows = []
    if system == "Lab":
        incs = pl.read_parquet(Path(cfg.lab_dir) / "incidents.parquet")
        for r in incs.iter_rows(named=True):
            a = int(np.searchsorted(ts, r["t_start"] - cfg.incident_pre_s * 1000, "left"))
            b = int(np.searchsorted(ts, r["t_end"] + cfg.incident_post_s * 1000, "right"))
            inc_rows.append((r["incident_id"], r["fault_type"], r["target_service"], r["split"], a, b))
        arrays["inc_a"] = np.array([x[4] for x in inc_rows], dtype=np.int64)
        arrays["inc_b"] = np.array([x[5] for x in inc_rows], dtype=np.int64)
        arrays["inc_meta"] = np.array([json.dumps(x[:4]) for x in inc_rows])
    out = Path(cfg.win_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{system}.npz", **arrays)
    by_split = {s: int(((wsplit == c)).sum()) for s, c in SPLIT_CODE.items()}
    pos = {s: int(wlabel[wsplit == c].sum()) for s, c in SPLIT_CODE.items()}
    return {"lines": len(uid), "windows": by_split, "positive_windows": pos}


def main(cfg: WindowsConfig) -> None:
    templates = build_template_table(cfg)
    Path(cfg.win_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.win_dir, "templates.json").write_text(
        json.dumps([{"svc": k[0], "level": k[1], "masked": k[2], "id": v}
                    for k, v in templates.items()]))
    root = Path(cfg.masked_dir)
    systems = cfg.systems or sorted(p.name for p in root.iterdir() if p.is_dir())
    for s in systems:
        print(s, build_system(s, cfg, templates), flush=True)


if __name__ == "__main__":
    args = parse_args("build windows")
    main(load_config(WindowsConfig, args.config))
