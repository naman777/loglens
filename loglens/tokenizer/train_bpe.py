"""Train byte-level BPE on masked, de-duplicated lines with a per-system cap.

Training text is ``<LVL:x> <SVC:sys> masked``. Duplicate masked lines are dropped before sampling
(templates repeat millions of times; BPE only needs each distinct line once). No system may exceed
``max_system_share`` of the sample, and the unseen system is excluded entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from loglens.config import load_config, parse_args
from loglens.seed import set_seed
from loglens.tokenizer.masking import level_token, service_token
from loglens.tokenizer.tok import DEFAULT_SERVICES, SPECIALS, structural_tokens


@dataclass
class TokenizerConfig:
    seed: int = 1337
    masked_dir: str = "data/masked"
    out_dir: str = "artifacts/tokenizer"
    vocab_sizes: list[int] = field(default_factory=lambda: [8000, 16000])
    sample_lines: int = 5_000_000
    max_system_share: float = 0.30
    unseen: str = "Thunderbird"
    services: list[str] = field(default_factory=lambda: list(DEFAULT_SERVICES))


def prefixed(level: str, svc: str, masked: str) -> str:
    return f"{level_token(level)} {service_token(svc, None)} {masked}"


def load_split_texts(masked_dir: str, system: str, split: str | None) -> list[str]:
    """Distinct prefixed texts of one system (optionally restricted to lines seen in ``split``)."""
    u = pl.read_parquet(Path(masked_dir) / system / "uniques.parquet")
    if split == "train":
        u = u.filter(pl.col("count_train") > 0)
    return [prefixed(lv, sv, m) for lv, sv, m in zip(u["level"], u["svc"], u["masked"], strict=True)]


def sample_training_texts(cfg: TokenizerConfig) -> list[str]:
    root = Path(cfg.masked_dir)
    systems = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != cfg.unseen)
    pools = {s: load_split_texts(cfg.masked_dir, s, "train") for s in systems}
    import random

    rng = random.Random(cfg.seed)
    # cap each system's share of the sample: iterate size = min(size, share * total) to a fixed point
    size = {s: len(p) for s, p in pools.items()}
    for _ in range(50):
        total = min(sum(size.values()), cfg.sample_lines)
        cap = int(cfg.max_system_share * total)
        new = {s: min(n, cap) for s, n in size.items()}
        if new == size:
            break
        size = new
    out: list[str] = []
    for s in systems:
        pool = pools[s]
        out.extend(rng.sample(pool, size[s]) if size[s] < len(pool) else pool)
    rng.shuffle(out)
    return out


def train_one(texts: list[str], vocab_size: int, services: list[str], path: Path) -> Tokenizer:
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    # lstrip absorbs the space before a mask/prefix token instead of spending a token on it
    special = [AddedToken(t, special=True, lstrip=True) for t in SPECIALS + structural_tokens(services)]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=special, min_frequency=2, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tk.train_from_iterator(texts, trainer=trainer)
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tk.token_to_id("[CLS]")), ("[SEP]", tk.token_to_id("[SEP]"))],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tk.save(str(path))
    return tk


def main(cfg: TokenizerConfig) -> None:
    set_seed(cfg.seed)
    texts = sample_training_texts(cfg)
    print(f"training on {len(texts):,} distinct masked lines", flush=True)
    for v in cfg.vocab_sizes:
        p = Path(cfg.out_dir) / f"loglens-bpe-{v // 1000}k.json"
        tk = train_one(texts, v, cfg.services, p)
        print(f"saved {p} (vocab {tk.get_vocab_size()})", flush=True)


if __name__ == "__main__":
    args = parse_args("train log BPE")
    main(load_config(TokenizerConfig, args.config))
