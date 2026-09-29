"""Window model: line embeddings + time-gap / service / level embeddings -> transformer -> heads."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from loglens.model.heads import AnomalyHead, SuspicionHead, TemplateHead


@dataclass
class WindowModelConfig:
    in_dim: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_window: int = 256
    n_gap_buckets: int = 16
    n_services: int = 32
    n_levels: int = 16
    n_templates: int = 2001  # top-2000 templates + "other"
    use_gap: bool = True


class WindowModel(nn.Module):
    """Input: emb (B,L,in_dim) float, gap/svc/lvl (B,L) long, pad (B,L) bool (True = padding).
    Output dict: ``anomaly`` (B,), ``suspicion`` (B,L), ``template_logits`` (B,L,T)."""

    def __init__(self, cfg: WindowModelConfig):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(cfg.in_dim, cfg.d_model)
        self.gap = nn.Embedding(cfg.n_gap_buckets, cfg.d_model)
        self.svc = nn.Embedding(cfg.n_services, cfg.d_model)
        self.lvl = nn.Embedding(cfg.n_levels, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_window + 1, cfg.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.mask_emb = nn.Parameter(torch.zeros(cfg.d_model))
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.anomaly_head = AnomalyHead(cfg.d_model)
        self.suspicion_head = SuspicionHead(cfg.d_model)
        self.template_head = TemplateHead(cfg.d_model, cfg.n_templates)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.mask_emb, std=0.02)

    def encode(self, emb, gap, svc, lvl, pad, line_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.inp(emb)
        if line_mask is not None:  # replace masked positions' content (keep context features)
            x = torch.where(line_mask.unsqueeze(-1), self.mask_emb.expand_as(x), x)
        if self.cfg.use_gap:
            x = x + self.gap(gap)
        x = x + self.svc(svc) + self.lvl(lvl)
        b, L, _ = x.shape
        x = x + self.pos(torch.arange(1, L + 1, device=x.device))
        x = torch.cat([self.cls.expand(b, 1, -1) + self.pos.weight[0], x], dim=1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        return self.norm(self.blocks(x, src_key_padding_mask=pad))

    def forward(self, emb, gap, svc, lvl, pad, line_mask=None, want_templates: bool = False) -> dict:
        h = self.encode(emb, gap, svc, lvl, pad, line_mask)
        out = {"anomaly": self.anomaly_head(h[:, 0]), "suspicion": self.suspicion_head(h[:, 1:])}
        if want_templates:
            out["template_logits"] = self.template_head(h[:, 1:])
        return out
