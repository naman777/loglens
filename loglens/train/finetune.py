"""Window-model training.

mode=unsup: masked-line template prediction on normal windows (no labels). Anomaly score =
            mean NLL over masked positions averaged over several random masks.
mode=sup:   anomaly BCE (pos-weighted) on labelled windows + suspicion BCE + pairwise margin on lab
            incident windows. Early stopping on val PR-AUC (anomaly) and RCA Recall@5 (suspicion).
Encoder is frozen (embeddings precomputed, Stage A). Thresholds are tuned on validation only.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from loglens.config import load_config, parse_args
from loglens.eval.metrics import evaluate_scores, pr_auc, rank_of_first_causal, rca_metrics
from loglens.model.window_model import WindowModel, WindowModelConfig
from loglens.seed import set_seed
from loglens.tracking import Tracker
from loglens.train.losses import anomaly_bce, pairwise_margin, suspicion_bce
from loglens.train.windata import MixedSampler, SystemData, load_systems, to_torch


@dataclass
class FinetuneConfig:
    seed: int = 1337
    mode: str = "sup"  # unsup | sup
    win_dir: str = "data/win"
    emb_dir: str = "data/emb"
    out_dir: str = "artifacts/window"
    init: str = ""  # checkpoint to start from (e.g. the unsupervised model)
    train_systems: list[str] = field(default_factory=lambda: ["BGL", "HDFS", "Lab"])
    unsup_systems: list[str] = field(default_factory=lambda: [
        "BGL", "HDFS", "Lab", "OpenStack", "Hadoop", "Spark", "Zookeeper", "Linux", "Apache", "HPC",
        "SSH", "Mac", "HealthApp", "Proxifier", "Android"])
    eval_systems: list[str] = field(default_factory=lambda: ["BGL", "HDFS", "Lab"])
    window: int = 256
    batch_size: int = 32
    steps: int = 3000
    eval_every: int = 250
    patience: int = 6
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 200
    grad_clip: float = 1.0
    sampler_power: float = 0.5
    line_mask_prob: float = 0.15
    rank_weight: float = 0.5
    susp_weight: float = 1.0
    use_gap: bool = True
    n_masks_eval: int = 4
    bf16: bool = True
    wandb: bool = False
    run_name: str = "finetune"
    n_templates: int = 2001
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    pretrained_encoder: bool = True  # bookkeeping for ablations only (embeddings picked via emb_dir)


def make_model(cfg: FinetuneConfig, n_services: int = 32) -> WindowModel:
    return WindowModel(WindowModelConfig(
        d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_ff=cfg.d_model * 4,
        n_templates=cfg.n_templates, use_gap=cfg.use_gap, max_window=cfg.window,
        n_services=n_services))


def _batch(model, b, dev, line_mask=None, want_templates=False):
    t = to_torch(b, dev)
    return t, model(t["emb"], t["gap"], t["svc"], t["lvl"], t["pad"], line_mask, want_templates)


def random_line_mask(pad: torch.Tensor, p: float) -> torch.Tensor:
    m = (torch.rand(pad.shape, device=pad.device) < p) & ~pad
    empty = ~m.any(1) & (~pad).any(1)
    if empty.any():
        pick = (torch.rand(pad.shape, device=pad.device) * (~pad)).argmax(1)
        rows = empty.nonzero().squeeze(1)
        m[rows, pick[rows]] = True
    return m


@torch.no_grad()
def unsup_scores(model, sd: SystemData, widx: np.ndarray, cfg: FinetuneConfig, dev: str,
                 bs: int = 64) -> np.ndarray:
    """Mean template NLL over masked lines, averaged over ``n_masks_eval`` masks."""
    model.eval()
    g = torch.Generator(device=dev).manual_seed(cfg.seed)
    out = np.zeros(len(widx), dtype=np.float64)
    for i in range(0, len(widx), bs):
        t = to_torch(sd.batch(widx[i: i + bs]), dev)
        nll = 0.0
        for _ in range(cfg.n_masks_eval):
            r = torch.rand(t["pad"].shape, device=dev, generator=g)
            m = (r < cfg.line_mask_prob) & ~t["pad"]
            empty = ~m.any(1) & (~t["pad"]).any(1)
            if empty.any():
                pick = (torch.rand(t["pad"].shape, device=dev, generator=g) * (~t["pad"])).argmax(1)
                rows = empty.nonzero().squeeze(1)
                m[rows, pick[rows]] = True
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=cfg.bf16 and dev == "cuda"):
                lg = model(t["emb"], t["gap"], t["svc"], t["lvl"], t["pad"], m, True)["template_logits"]
            nl = F.cross_entropy(lg.float().transpose(1, 2), t["tid"], reduction="none")
            nll = nll + (nl * m).sum(1) / m.sum(1).clamp(min=1)
        out[i: i + bs] = (nll / cfg.n_masks_eval).cpu().numpy()
    return out


@torch.no_grad()
def sup_scores(model, sd: SystemData, widx: np.ndarray, cfg: FinetuneConfig, dev: str,
               bs: int = 64) -> np.ndarray:
    model.eval()
    out = np.zeros(len(widx), dtype=np.float64)
    for i in range(0, len(widx), bs):
        t = to_torch(sd.batch(widx[i: i + bs]), dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=cfg.bf16 and dev == "cuda"):
            a = model(t["emb"], t["gap"], t["svc"], t["lvl"], t["pad"])["anomaly"]
        out[i: i + bs] = a.float().cpu().numpy()
    return out


@torch.no_grad()
def rank_incident(model, sd: SystemData, a: int, b: int, cfg: FinetuneConfig, dev: str):
    """Sliding windows over stream slice [a, b); per-line suspicion = max over covering windows."""
    model.eval()
    spans, s = [], a
    while True:
        e = min(s + cfg.window, b)
        spans.append((s, e))
        if e >= b:
            break
        s += cfg.window // 2
    score = np.full(b - a, -1e9, dtype=np.float64)
    for i in range(0, len(spans), 16):
        chunk = spans[i: i + 16]
        t = to_torch(sd.slice_batch(chunk), dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=cfg.bf16 and dev == "cuda"):
            sl = model(t["emb"], t["gap"], t["svc"], t["lvl"], t["pad"])["suspicion"].float().cpu().numpy()
        for (x, y), row in zip(chunk, sl, strict=True):
            score[x - a: y - a] = np.maximum(score[x - a: y - a], row[: y - x])
    return score


def rca_eval(model, sd: SystemData, split: str, cfg: FinetuneConfig, dev: str) -> dict:
    ranks, by_type = [], {}
    for k, (iid, ftype, tgt, sp) in enumerate(sd.incidents or []):
        if sp != split:
            continue
        a, b = int(sd.z["inc_a"][k]), int(sd.z["inc_b"][k])
        sc = rank_incident(model, sd, a, b, cfg, dev)
        r = rank_of_first_causal(sc, sd.z["causal"][a:b])
        ranks.append(r)
        by_type.setdefault(ftype, []).append(r)
    res = rca_metrics(ranks)
    res["by_fault_type"] = {t: rca_metrics(r) for t, r in by_type.items()}
    return res


def train(cfg: FinetuneConfig) -> dict:
    set_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sysnames = sorted(set(cfg.train_systems if cfg.mode == "sup" else cfg.unsup_systems)
                      | set(cfg.eval_systems))
    systems = load_systems(sysnames, cfg.win_dir, cfg.emb_dir)
    model = make_model(cfg).to(dev)
    if cfg.init:
        st = torch.load(cfg.init, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"], strict=False)
    rng = np.random.default_rng(cfg.seed)
    tr = cfg.train_systems if cfg.mode == "sup" else cfg.unsup_systems
    labeled = "any" if cfg.mode == "sup" else "normal"
    sampler = MixedSampler({n: systems[n] for n in tr}, "train", labeled, cfg.sampler_power, rng)
    # class balance for the anomaly loss
    ys = np.concatenate([systems[n].z["wlabel"][systems[n].windows("train")] for n in tr])
    pos_weight = float((ys == 0).sum() / max(ys.sum(), 1)) ** 0.5
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tracker = Tracker("loglens", cfg.run_name, cfg, manifest="data/splits/manifest.json",
                      use_wandb=cfg.wandb)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, bad, t0 = -1e9, 0, time.time()
    for step in range(1, cfg.steps + 1):
        lr = cfg.lr * min(step / cfg.warmup, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        name, b = sampler.sample(cfg.batch_size)
        t = to_torch(b, dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=cfg.bf16 and dev == "cuda"):
            if cfg.mode == "unsup":
                m = random_line_mask(t["pad"], cfg.line_mask_prob)
                o = model(t["emb"], t["gap"], t["svc"], t["lvl"], t["pad"], m, True)
                nl = F.cross_entropy(o["template_logits"].float().transpose(1, 2), t["tid"],
                                    reduction="none")
                loss = (nl * m).sum() / m.sum().clamp(min=1)
            else:
                o = model(t["emb"], t["gap"], t["svc"], t["lvl"], t["pad"])
                loss = anomaly_bce(o["anomaly"].float(), t["y"], pos_weight)
                if t["causal"].sum() > 0:
                    sl = o["suspicion"].float()
                    loss = loss + cfg.susp_weight * suspicion_bce(sl, t["causal"], t["pad"]) \
                        + cfg.rank_weight * pairwise_margin(sl, t["causal"], t["pad"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % 50 == 0:
            tracker.log({"loss": loss.item(), "lr": lr}, step=step)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            metric, info = validate(model, systems, cfg, dev)
            tracker.log({"val_metric": metric, **{k: v for k, v in info.items()
                                                  if isinstance(v, float)}}, step=step)
            print(f"step {step} loss {loss.item():.4f} val {metric:.4f} {info} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if metric > best:
                best, bad = metric, 0
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "step": step}, out / "best.pt")
            else:
                bad += 1
                if bad >= cfg.patience:
                    print("early stop", flush=True)
                    break
    tracker.finish()
    return {"best_val": best, "out": str(out / "best.pt")}


def validate(model, systems, cfg: FinetuneConfig, dev: str) -> tuple[float, dict]:
    """Higher is better: mean val PR-AUC over labelled eval systems (+ lab RCA Recall@5)."""
    info, vals = {}, []
    for n in cfg.eval_systems:
        sd = systems[n]
        w = sd.windows("val")
        y = sd.z["wlabel"][w]
        if y.sum() == 0:
            continue
        s = (unsup_scores if cfg.mode == "unsup" else sup_scores)(model, sd, w, cfg, dev)
        ap = pr_auc(y, s)
        info[f"prauc_{n}"] = round(float(ap), 4)
        vals.append(ap)
        if cfg.mode == "sup" and sd.incidents:
            rca = rca_eval(model, sd, "val", cfg, dev)
            info["rca_r5_val"] = round(float(rca["recall@5"]), 4)
            vals.append(rca["recall@5"])
    if cfg.mode == "unsup" and not vals:
        return -1e9, info
    return float(np.mean(vals)), info


if __name__ == "__main__":
    args = parse_args("window model training")
    print(train(load_config(FinetuneConfig, args.config)))
    _ = evaluate_scores
