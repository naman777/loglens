"""Losses for the window model."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def anomaly_bce(logit: torch.Tensor, y: torch.Tensor, pos_weight: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logit, y, pos_weight=torch.tensor(pos_weight, device=logit.device))


def suspicion_bce(logit: torch.Tensor, causal: torch.Tensor, pad: torch.Tensor,
                  pos_weight: float = 10.0) -> torch.Tensor:
    valid = ~pad
    bce = F.binary_cross_entropy_with_logits(
        logit, causal, pos_weight=torch.tensor(pos_weight, device=logit.device), reduction="none")
    return (bce * valid).sum() / valid.sum().clamp(min=1)


def pairwise_margin(logit: torch.Tensor, causal: torch.Tensor, pad: torch.Tensor,
                    margin: float = 1.0, n_neg: int = 32) -> torch.Tensor:
    """Every causal line should outrank sampled non-causal lines by ``margin`` (per window)."""
    loss, cnt = logit.new_zeros(()), 0
    for i in range(logit.size(0)):
        pos = logit[i][(causal[i] > 0) & ~pad[i]]
        negm = (causal[i] == 0) & ~pad[i]
        if pos.numel() == 0 or negm.sum() == 0:
            continue
        neg_all = logit[i][negm]
        k = min(n_neg, neg_all.numel())
        neg = neg_all.topk(k).values  # hardest negatives
        loss = loss + F.relu(margin - (pos[:, None] - neg[None, :])).mean()
        cnt += 1
    return loss / max(cnt, 1)
