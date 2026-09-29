"""Pre-LN transformer line encoder: token ids -> 256-d line embedding, with a tied MLM head."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LineEncoderConfig:
    vocab_size: int = 16000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 128
    embed_dim: int = 256
    pad_id: int = 0


class LineEncoder(nn.Module):
    def __init__(self, cfg: LineEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))  # output layer tied to self.tok
        self.proj = nn.Linear(cfg.d_model, cfg.embed_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        pad = ids == self.cfg.pad_id
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.drop(self.tok(ids) + self.pos(pos))
        x = self.blocks(x, src_key_padding_mask=pad)
        return self.norm(x)

    def mlm_logits(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.tok.weight, self.mlm_bias)

    def embed(self, ids: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """Mean over non-pad tokens, then linear projection to ``embed_dim``."""
        h = self.hidden(ids) if h is None else h
        m = (ids != self.cfg.pad_id).unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.proj(pooled)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ids)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
