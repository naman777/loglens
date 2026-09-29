"""MLM pretraining of the line encoder on pre-tokenized unique masked lines.

Sampling is over *distinct* masked lines (weights: equal per line inside a system, no system above
``max_system_share``), so common templates are not seen millions of times. Optional SimCSE-style
contrastive loss (same line, two dropout masks).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from loglens.config import load_config, parse_args
from loglens.model.line_encoder import LineEncoder, LineEncoderConfig, count_params
from loglens.seed import set_seed
from loglens.tracking import Tracker

N_SPECIAL = 5  # [PAD] [CLS] [SEP] [MASK] [UNK]
PAD, CLS, SEP, MASK = 0, 1, 2, 3


@dataclass
class PretrainConfig:
    seed: int = 1337
    tok_dir: str = "data/tok"
    out_dir: str = "artifacts/pretrain"
    tokenizer: str = "artifacts/tokenizer/loglens-bpe-16k.json"
    unseen: str = "Thunderbird"
    exclude: list[str] = field(default_factory=list)
    max_system_share: float = 0.30
    # model
    vocab_size: int = 16000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 128
    embed_dim: int = 256
    # optimisation
    lr: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 2000
    min_lr_frac: float = 0.1
    batch_size: int = 256
    max_batch_tokens: int = 16384  # rows * padded width cap (keeps a 6 GB GPU out of shared memory)
    max_steps: int = 50_000
    grad_clip: float = 1.0
    mask_prob: float = 0.15
    simcse_weight: float = 0.0
    simcse_rows: int = 64  # contrastive loss on the first N rows only (memory)
    bf16: bool = True
    # bookkeeping
    log_every: int = 50
    eval_every: int = 1000
    ckpt_every_min: float = 30.0
    keep_ckpts: int = 3
    val_frac: float = 0.01
    max_minutes: float = 0.0  # wall-clock budget; 0 = none (cosine is still tied to max_steps)
    wandb: bool = False
    run_name: str = "pretrain"


class Corpus:
    """Ragged token store over distinct lines from several systems."""

    def __init__(self, cfg: PretrainConfig):
        ids, offs, sysid, self.names = [], [], [], []
        base = 0
        for f in sorted(Path(cfg.tok_dir).glob("*.npz")):
            name = f.stem
            if name == cfg.unseen or name in cfg.exclude:
                continue
            z = np.load(f)
            keep = np.nonzero(z["count_train"] > 0)[0]
            if len(keep) == 0:
                continue
            o = z["offsets"]
            starts, ends = o[:-1][keep], o[1:][keep]
            lens = ends - starts
            flat = np.concatenate([z["ids"][s:e] for s, e in zip(starts, ends, strict=True)]) \
                if len(keep) < 5000 else self._gather(z["ids"], starts, lens)
            ids.append(flat)
            offs.append(np.concatenate([[0], np.cumsum(lens)]) + base)
            base += int(lens.sum())
            sysid.append(np.full(len(keep), len(self.names)))
            self.names.append(name)
        self.ids = np.concatenate(ids)
        self.starts = np.concatenate([o[:-1] for o in offs])
        self.lens = np.concatenate([np.diff(o) for o in offs])
        self.system = np.concatenate(sysid)
        rng = np.random.default_rng(cfg.seed)
        self.is_val = rng.random(len(self.lens)) < cfg.val_frac
        train_idx = np.nonzero(~self.is_val)[0]
        self.train_idx = train_idx
        self.weights = self._weights(train_idx, cfg.max_system_share)
        self.val_idx = np.nonzero(self.is_val)[0]
        self.max_len = cfg.max_len

    @staticmethod
    def _gather(ids: np.ndarray, starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
        total = int(lens.sum())
        out = np.empty(total, dtype=ids.dtype)
        pos = np.concatenate([[0], np.cumsum(lens)])
        # vectorised ragged gather
        idx = np.repeat(starts - pos[:-1], lens) + np.arange(total)
        out[:] = ids[idx]
        return out

    def _weights(self, idx: np.ndarray, cap: float) -> np.ndarray:
        counts = np.bincount(self.system[idx], minlength=len(self.names)).astype(np.float64)
        share = counts / counts.sum()
        for _ in range(20):  # water-fill: cap shares, redistribute the excess
            over = share > cap
            if not over.any():
                break
            excess = (share[over] - cap).sum()
            share[over] = cap
            free = ~over & (share > 0)
            share[free] += excess * share[free] / share[free].sum()
        w = share[self.system[idx]] / counts[self.system[idx]]
        return w / w.sum()

    def batch(self, rng: np.random.Generator, indices: np.ndarray) -> np.ndarray:
        lens = np.minimum(self.lens[indices], self.max_len)
        out = np.zeros((len(indices), int(lens.max())), dtype=np.int64)
        for r, (i, ln) in enumerate(zip(indices, lens, strict=True)):
            s = self.starts[i]
            out[r, :ln] = self.ids[s: s + ln]
        return out

    def train_batches(self, rng: np.random.Generator, bs: int, max_tokens: int, pool_batches: int = 32):
        while True:
            pool = rng.choice(self.train_idx, size=bs * pool_batches, p=self.weights)
            pool = pool[np.argsort(self.lens[pool], kind="stable")]
            chunks, cur = [], []
            for i in pool:
                width = min(int(self.lens[i]), self.max_len)
                if cur and (len(cur) >= bs or (len(cur) + 1) * width > max_tokens):
                    chunks.append(np.array(cur))
                    cur = []
                cur.append(i)
            if cur:
                chunks.append(np.array(cur))
            rng.shuffle(chunks)
            yield from (self.batch(rng, c) for c in chunks)


def mlm_mask(ids: torch.Tensor, vocab: int, p: float, gen: torch.Generator | None = None):
    """15% of non-special tokens; 80% [MASK], 10% random, 10% unchanged. Returns (inputs, labels)."""
    cand = ids >= N_SPECIAL
    r = torch.rand(ids.shape, device=ids.device, generator=gen)
    chosen = cand & (r < p)
    # guarantee at least one masked token per row that has candidates
    empty = ~chosen.any(1) & cand.any(1)
    if empty.any():
        rows = empty.nonzero().squeeze(1)
        pick = (torch.rand(ids.shape, device=ids.device) * cand).argmax(1)
        chosen[rows, pick[rows]] = True
    labels = torch.where(chosen, ids, torch.full_like(ids, -100))
    x = ids.clone()
    act = torch.rand(ids.shape, device=ids.device, generator=gen)
    x[chosen & (act < 0.8)] = MASK
    rnd = chosen & (act >= 0.8) & (act < 0.9)
    x[rnd] = torch.randint(N_SPECIAL, vocab, (int(rnd.sum()),), device=ids.device)
    return x, labels


def mlm_loss(model: LineEncoder, ids: torch.Tensor, labels_ids: torch.Tensor, cfg: PretrainConfig):
    h = model.hidden(ids)
    sel = labels_ids != -100
    logits = model.mlm_logits(h[sel])
    return F.cross_entropy(logits.float(), labels_ids[sel]), h


def lr_at(step: int, cfg: PretrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    prog = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    prog = min(prog, 1.0)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(model: LineEncoder, corpus: Corpus, cfg: PretrainConfig, dev: str, n_batches: int = 20):
    model.eval()
    gen = torch.Generator(device=dev).manual_seed(0)
    rng = np.random.default_rng(0)
    idx = corpus.val_idx
    tot = 0.0
    for b in range(n_batches):
        pick = rng.choice(idx, size=min(cfg.batch_size, len(idx)), replace=False)
        ids = torch.from_numpy(corpus.batch(rng, pick)).to(dev)
        x, y = mlm_mask(ids, cfg.vocab_size, cfg.mask_prob, gen)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=cfg.bf16 and dev == "cuda"):
            loss, _ = mlm_loss(model, x, y, cfg)
        tot += loss.item()
    model.train()
    return tot / n_batches


def save_ckpt(model, opt, step, cfg, path: Path, keep: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "cfg": asdict(cfg)}, path)
    ck = sorted(path.parent.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    for old in ck[:-keep]:
        old.unlink()


def main(cfg: PretrainConfig) -> None:
    set_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = Corpus(cfg)
    print(f"corpus: {len(corpus.lens):,} distinct lines, {corpus.lens.sum():,} tokens, "
          f"systems={corpus.names}, val={len(corpus.val_idx):,}", flush=True)
    mcfg = LineEncoderConfig(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff,
                             cfg.dropout, cfg.max_len, cfg.embed_dim)
    model = LineEncoder(mcfg).to(dev)
    print(f"params: {count_params(model):,} on {dev}", flush=True)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=cfg.betas)
    out = Path(cfg.out_dir)
    step = 0
    ck = sorted(out.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    if ck:
        st = torch.load(ck[-1], map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"resumed from {ck[-1]} @ step {step}", flush=True)
    tracker = Tracker("loglens", cfg.run_name, cfg, use_wandb=cfg.wandb)
    rng = np.random.default_rng(cfg.seed + step)
    batches = corpus.train_batches(rng, cfg.batch_size, cfg.max_batch_tokens)
    model.train()
    t0 = t_ck = time.time()
    tok_seen, ema = 0, None
    while step < cfg.max_steps:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        ids = torch.from_numpy(next(batches)).to(dev, non_blocking=True)
        x, y = mlm_mask(ids, cfg.vocab_size, cfg.mask_prob)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=cfg.bf16 and dev == "cuda"):
            loss, _ = mlm_loss(model, x, y, cfg)
            total = loss
            if cfg.simcse_weight > 0:
                sub = ids[: cfg.simcse_rows]
                z1, z2 = model.embed(sub), model.embed(sub)
                sim = F.normalize(z1.float(), dim=-1) @ F.normalize(z2.float(), dim=-1).T / 0.05
                total = loss + cfg.simcse_weight * F.cross_entropy(
                    sim, torch.arange(len(sub), device=dev))
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        tok_seen += int((ids != PAD).sum())
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % cfg.log_every == 0:
            el = time.time() - t0
            tracker.log({"loss": loss.item(), "loss_ema": ema, "lr": lr, "grad_norm": float(gn),
                         "tokens_per_s": tok_seen / el}, step=step)
            print(f"step {step} loss {loss.item():.4f} ema {ema:.4f} lr {lr:.2e} gn {float(gn):.2f} "
                  f"tok/s {tok_seen / el:,.0f}", flush=True)
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            v = evaluate(model, corpus, cfg, dev)
            tracker.log({"val_loss": v}, step=step)
            print(f"  val_loss {v:.4f}", flush=True)
        if (time.time() - t_ck) / 60 >= cfg.ckpt_every_min:
            save_ckpt(model, opt, step, cfg, out / f"step-{step}.pt", cfg.keep_ckpts)
            t_ck = time.time()
        if cfg.max_minutes and (time.time() - t0) / 60 >= cfg.max_minutes:
            print("wall-clock budget reached", flush=True)
            break
    save_ckpt(model, opt, step, cfg, out / f"step-{step}.pt", cfg.keep_ckpts)
    torch.save({"model": model.state_dict(), "cfg": asdict(mcfg)}, out / "encoder.pt")
    tracker.finish()


if __name__ == "__main__":
    args = parse_args("line encoder MLM pretraining")
    main(load_config(PretrainConfig, args.config))
