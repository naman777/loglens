"""Tokenizer comparison report: tokens/line, [UNK] rate, share of lines > 128 tokens.

Compared on held-out test lines (all systems' test split, plus the unseen system):
raw GPT-2, GPT-2 on masked text, LogLens 8k, LogLens 16k.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from tokenizers import Tokenizer

from loglens.config import load_config, parse_args
from loglens.data.build import DataConfig
from loglens.seed import set_seed
from loglens.tokenizer.masking import level_token, service_token


@dataclass
class TokEvalConfig:
    seed: int = 1337
    parquet_dir: str = "data/parquet"
    masked_dir: str = "data/masked"
    tokenizer_dir: str = "artifacts/tokenizer"
    gpt2_path: str = "artifacts/gpt2_tokenizer.json"
    out_md: str = "docs/tokenizer_report.md"
    sample_per_system: int = 20000
    unseen: str = "Thunderbird"
    vocab_sizes: list[int] = field(default_factory=lambda: [8000, 16000])
    max_len: int = 128


def test_lines(cfg: TokEvalConfig, system: str, rng: random.Random) -> tuple[list[str], list[str]]:
    """Raw and masked+prefixed test lines, sampled *by line* (so frequent templates weigh in)."""
    lines = pl.read_parquet(Path(cfg.masked_dir) / system / "lines.parquet").filter(
        pl.col("split") == "test")
    u = pl.read_parquet(Path(cfg.masked_dir) / system / "uniques.parquet")
    n = min(cfg.sample_per_system, len(lines))
    uids = lines["uid"].to_list()
    pick = rng.sample(uids, n) if n < len(uids) else uids
    ud = dict(zip(u["uid"], zip(u["level"], u["svc"], u["masked"], strict=True), strict=True))
    masked = [f"{level_token(ud[i][0])} {service_token(ud[i][1], None)} {ud[i][2]}" for i in pick]
    # raw text: best available proxy is the masked text's source message via a second read
    raw_df = pl.read_parquet(Path(cfg.parquet_dir) / system / "part-*.parquet",
                             columns=["message", "split"]).filter(pl.col("split") == "test")
    raw = raw_df["message"].sample(n, seed=cfg.seed).to_list()
    return raw, masked


def stats(tk: Tokenizer, texts: list[str], max_len: int) -> dict:
    tk.no_padding()
    tk.no_truncation()
    enc = tk.encode_batch(texts)
    lens = np.array([len(e.ids) for e in enc])
    unk = tk.token_to_id("[UNK]")
    unk_rate = 0.0
    if unk is not None:
        unk_rate = float(np.mean([unk in e.ids for e in enc]))
    return {"mean": float(lens.mean()), "p95": float(np.percentile(lens, 95)),
            "unk": unk_rate, "over": float((lens > max_len).mean())}


def main(cfg: TokEvalConfig) -> str:
    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    gpt2 = Tokenizer.from_file(cfg.gpt2_path)
    lens_tk = {v: Tokenizer.from_file(str(Path(cfg.tokenizer_dir) / f"loglens-bpe-{v // 1000}k.json"))
               for v in cfg.vocab_sizes}
    systems = sorted(p.name for p in Path(cfg.masked_dir).iterdir() if p.is_dir())
    rows = []
    md = ["# Tokenizer report", "",
          "Tokens per line on held-out **test** lines (sampled by line, so frequent templates weigh "
          "in). LogLens counts include `[CLS]`/`[SEP]` and the level/service prefix tokens. "
          f"`{cfg.unseen}` is the unseen system: excluded from BPE training.", "",
          "| system | GPT-2 raw (mean / p95) | GPT-2 masked (mean) | " + " | ".join(
              f"LogLens {v // 1000}k (mean / p95 / UNK% / >128%)" for v in cfg.vocab_sizes)
          + " | ratio raw-GPT2 / LogLens-8k |", "|" + " --- |" * (4 + len(cfg.vocab_sizes))]
    agg: dict[str, list[float]] = {}
    for s in systems:
        raw, masked = test_lines(cfg, s, rng)
        g_raw = stats(gpt2, raw, cfg.max_len)
        g_mask = stats(gpt2, masked, cfg.max_len)
        cells, first = [], None
        for v, tk in lens_tk.items():
            st = stats(tk, masked, cfg.max_len)
            first = first or st
            cells.append(f"{st['mean']:.1f} / {st['p95']:.0f} / {st['unk'] * 100:.2f} / "
                         f"{st['over'] * 100:.2f}")
            agg.setdefault(f"{v}_mean", []).append(st["mean"])
        ratio = g_raw["mean"] / first["mean"]
        rows.append((s, ratio))
        star = " (unseen)" if s == cfg.unseen else ""
        md.append(f"| {s}{star} | {g_raw['mean']:.1f} / {g_raw['p95']:.0f} | {g_mask['mean']:.1f} | "
                  + " | ".join(cells) + f" | {ratio:.2f}x |")
    md += ["", f"Mean ratio across systems: {np.mean([r for _, r in rows]):.2f}x fewer tokens "
           "(raw GPT-2 / LogLens 8k).", ""]
    text = "\n".join(md)
    Path(cfg.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_md).write_text(text, encoding="utf-8")
    print(text)
    return text


if __name__ == "__main__":
    args = parse_args("tokenizer comparison")
    main(load_config(TokEvalConfig, args.config))
    _ = DataConfig  # keep import for config discovery
