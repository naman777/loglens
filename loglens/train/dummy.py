"""Phase 0 smoke run: 2-layer MLP on random data, loss logged through the tracker."""
from __future__ import annotations

from dataclasses import dataclass

from loglens.config import load_config, parse_args
from loglens.seed import set_seed
from loglens.tracking import Tracker


@dataclass
class DummyConfig:
    seed: int = 1337
    steps: int = 200
    lr: float = 1e-2
    wandb: bool = False


def run(cfg: DummyConfig) -> list[float]:
    import torch
    from torch import nn

    set_seed(cfg.seed)
    tracker = Tracker("loglens", "dummy", cfg, use_wandb=cfg.wandb)
    x = torch.randn(512, 16)
    y = (x[:, :4].sum(1, keepdim=True) > 0).float()
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    losses = []
    for step in range(cfg.steps):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 20 == 0:
            tracker.log({"loss": loss.item()}, step=step)
    tracker.finish()
    return losses


if __name__ == "__main__":
    args = parse_args("dummy training run")
    losses = run(load_config(DummyConfig, args.config))
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
