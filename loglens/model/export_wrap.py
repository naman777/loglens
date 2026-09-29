"""ONNX-friendly re-implementations of the two models' forward passes.

``nn.MultiheadAttention`` bakes shapes into Reshape nodes under tracing and breaks dynamic
batch/sequence axes, so for export we recompute the same math functionally from the trained
parameters (no retraining, weights are shared with the PyTorch modules).
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _layer(layer: nn.TransformerEncoderLayer, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Pre-LN block. ``bias``: (B,1,1,L) additive attention bias (large negative on padding)."""
    at = layer.self_attn
    H = at.num_heads
    h = layer.norm1(x)
    B, L, D = h.shape
    q, k, v = F.linear(h, at.in_proj_weight, at.in_proj_bias).chunk(3, dim=-1)
    dh = D // H
    q = q.reshape(B, L, H, dh).transpose(1, 2)
    k = k.reshape(B, L, H, dh).transpose(1, 2)
    v = v.reshape(B, L, H, dh).transpose(1, 2)
    a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh) + bias, dim=-1) @ v
    a = a.transpose(1, 2).reshape(B, L, D)
    x = x + F.linear(a, at.out_proj.weight, at.out_proj.bias)
    h = layer.norm2(x)
    return x + layer.linear2(F.gelu(layer.linear1(h)))


def _pad_bias(pad: torch.Tensor) -> torch.Tensor:
    return pad[:, None, None, :].to(torch.float32) * -1e4


class EncoderExport(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        m = self.m
        pad = ids == m.cfg.pad_id
        pos = torch.arange(ids.size(1), device=ids.device)
        x = m.tok(ids) + m.pos(pos)
        bias = _pad_bias(pad)
        for layer in m.blocks.layers:
            x = _layer(layer, x, bias)
        h = m.norm(x)
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return m.proj(pooled)


class WindowExport(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, emb, gap, svc, lvl, pad):
        m = self.m
        x = m.inp(emb)
        if m.cfg.use_gap:
            x = x + m.gap(gap)
        x = x + m.svc(svc) + m.lvl(lvl)
        b = x.size(0)
        L = x.size(1)
        x = x + m.pos(torch.arange(1, L + 1, device=x.device))
        cls = m.cls.expand(b, 1, -1) + m.pos.weight[0]
        x = torch.cat([cls, x], dim=1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        bias = _pad_bias(pad)
        for layer in m.blocks.layers:
            x = _layer(layer, x, bias)
        h = m.norm(x)
        return m.anomaly_head(h[:, 0]), m.suspicion_head(h[:, 1:])
