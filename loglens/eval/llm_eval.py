"""LLM baselines: same prompt and JSON schema for a small open LLM (local, transformers) and a
frontier API model. Every response is cached on disk keyed by a hash of (model, prompt) so reruns are
free; tokens in/out are recorded per call and cost = tokens x the published price (pass the price and
the date you ran it; they go in the table footnote).

The frontier provider needs an API key (env ANTHROPIC_API_KEY) and is *skipped, not faked* when
absent.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from loglens.eval.metrics import rank_of_first_causal
from loglens.tokenizer.masking import LEVELS
from loglens.train.windata import SystemData

PROMPT = """You are an SRE log triage assistant. Below is a window of {n} log lines (masked: numbers, \
ids and addresses replaced by placeholders). Decide whether the window contains an incident and \
which lines best explain it.

Return ONLY JSON: {{"anomaly": true|false, "confidence": 0..1, "suspect_lines": [up to 5 line \
numbers, most suspicious first]}}

LOG WINDOW:
{lines}
"""


@dataclass
class LLMConfig:
    provider: str = "hf_local"  # hf_local | anthropic
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    cache_dir: str = "results/llm_cache"
    max_prompt_tokens: int = 3500
    max_new_tokens: int = 96
    n_anomaly_windows: int = 300
    seed: int = 1337
    price_in_per_mtok: float = 0.0  # published $/1M input tokens on the run date
    price_out_per_mtok: float = 0.0
    price_date: str = ""
    masked_dir: str = "data/masked"
    win_dir: str = "data/win"
    emb_dir: str = "data/emb"
    systems: list[str] = field(default_factory=lambda: ["BGL"])


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()[:24]


class Provider:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.calls = self.tokens_in = self.tokens_out = self.cache_hits = 0
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        self._impl = None

    def _load(self):
        if self._impl is not None:
            return
        if self.cfg.provider == "hf_local":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.cfg.model)
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            self._impl = (tok, model)
        elif self.cfg.provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set: frontier evaluation skipped")
            import anthropic

            self._impl = anthropic.Anthropic()
        else:
            raise ValueError(self.cfg.provider)

    def count_tokens(self, text: str) -> int:
        self._load()
        if self.cfg.provider == "hf_local":
            return len(self._impl[0](text).input_ids)
        return len(text) // 3  # rough; the API reports exact usage per call

    def complete(self, prompt: str) -> dict:
        key = cache_key(self.cfg.model, prompt)
        f = Path(self.cfg.cache_dir) / f"{key}.json"
        if f.exists():
            self.cache_hits += 1
            return json.loads(f.read_text())
        self._load()
        if self.cfg.provider == "hf_local":
            import torch

            tok, model = self._impl
            msgs = [{"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
            n_in = enc["input_ids"].shape[1]
            text = tok.decode(out[0, n_in:], skip_special_tokens=True)
            rec = {"text": text, "tokens_in": int(n_in), "tokens_out": int(out.shape[1] - n_in)}
        else:
            r = self._impl.messages.create(model=self.cfg.model, max_tokens=self.cfg.max_new_tokens,
                                           temperature=0, messages=[{"role": "user", "content": prompt}])
            rec = {"text": r.content[0].text, "tokens_in": r.usage.input_tokens,
                   "tokens_out": r.usage.output_tokens}
        f.write_text(json.dumps(rec))
        self.calls += 1
        self.tokens_in += rec["tokens_in"]
        self.tokens_out += rec["tokens_out"]
        return rec

    def cost(self) -> dict:
        usd = (self.tokens_in * self.cfg.price_in_per_mtok + self.tokens_out * self.cfg.price_out_per_mtok) / 1e6
        return {"calls": self.calls, "cache_hits": self.cache_hits, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "usd": usd, "price_date": self.cfg.price_date}


def parse_response(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    sus = [int(x) for x in d.get("suspect_lines", []) if isinstance(x, (int, float, str)) and str(x).lstrip("-").isdigit()]
    return {"anomaly": bool(d.get("anomaly", False)), "confidence": float(d.get("confidence", 0.5) or 0.5),
            "suspects": sus[:5], "valid": bool(d)}


def render_window(sd: SystemData, uniq: pl.DataFrame, a: int, b: int, provider: Provider,
                  max_tokens: int, from_end: bool = False) -> tuple[str, int, int]:
    """Numbered masked lines; truncated (keep the start, or the end for incident slices) to the
    token budget. Returns (prompt, n_lines_shown, n_lines_total)."""
    masked = uniq["masked"].to_list()
    level = uniq["level"].to_list()
    svc = uniq["svc"].to_list()
    uids = sd.z["uid"][a:b]
    lines = [f"{i + 1} [{level[u]}] {svc[u]}: {masked[u]}" for i, u in enumerate(uids)]
    total = len(lines)
    budget = max_tokens - 200
    shown, used = [], 0
    seq = reversed(list(enumerate(lines))) if from_end else enumerate(lines)
    for i, ln in seq:
        t = provider.count_tokens(ln) + 1
        if used + t > budget:
            break
        shown.append((i, ln))
        used += t
    shown.sort()
    return PROMPT.format(n=total, lines="\n".join(ln for _, ln in shown)), len(shown), total


def eval_anomaly(cfg: LLMConfig, provider: Provider, system: str) -> dict:
    """Anomaly yes/no on a fixed, seeded, class-balanced sample of test windows."""
    sd = SystemData(system, cfg.win_dir, cfg.emb_dir)
    uniq = pl.read_parquet(Path(cfg.masked_dir) / system / "uniques.parquet").sort("uid")
    w = sd.windows("test")
    y = sd.z["wlabel"][w]
    rng = np.random.default_rng(cfg.seed)
    half = cfg.n_anomaly_windows // 2
    pos = rng.choice(w[y == 1], min(half, int((y == 1).sum())), replace=False)
    neg = rng.choice(w[y == 0], min(half, int((y == 0).sum())), replace=False)
    picks = np.concatenate([pos, neg])
    truth, pred, conf, trunc, invalid = [], [], [], 0, 0
    for wi in picks:
        a, b = sd.span(int(wi))
        prompt, shown, total = render_window(sd, uniq, a, b, provider, cfg.max_prompt_tokens)
        trunc += shown < total
        r = parse_response(provider.complete(prompt)["text"])
        invalid += not r["valid"]
        truth.append(int(sd.z["wlabel"][wi]))
        pred.append(int(r["anomaly"]))
        conf.append(r["confidence"] if r["anomaly"] else 1 - r["confidence"])
    from loglens.eval.metrics import pr_auc, prf

    return {**prf(np.array(truth), np.array(pred)), "pr_auc": pr_auc(np.array(truth), np.array(conf)),
            "n": len(picks), "truncated_windows": int(trunc), "invalid_json": int(invalid),
            "note": "class-balanced 50/50 sample: F1 not comparable to natural-rate tables"}


def eval_rca(cfg: LLMConfig, provider: Provider) -> dict:
    sd = SystemData("Lab", cfg.win_dir, cfg.emb_dir)
    uniq = pl.read_parquet(Path(cfg.masked_dir) / "Lab" / "uniques.parquet").sort("uid")
    ranks, trunc, shown_tot, total_tot = [], 0, 0, 0
    for k, (iid, ftype, tgt, sp) in enumerate(sd.incidents):
        if sp != "test":
            continue
        a, b = int(sd.z["inc_a"][k]), int(sd.z["inc_b"][k])
        prompt, shown, total = render_window(sd, uniq, a, b, provider, cfg.max_prompt_tokens)
        trunc += shown < total
        shown_tot += shown
        total_tot += total
        r = parse_response(provider.complete(prompt)["text"])
        score = np.zeros(b - a)
        for rank, ln in enumerate(r["suspects"]):
            if 1 <= ln <= b - a:
                score[ln - 1] = 10 - rank
        causal = sd.z["causal"][a:b]
        # a line the LLM never mentions ties with the rest at 0 -> pessimistic tie rule applies
        ranks.append(rank_of_first_causal(score, causal))
    from loglens.eval.metrics import rca_metrics

    return {**rca_metrics(ranks), "truncated_incidents": trunc, "mean_lines_shown_frac":
            shown_tot / max(total_tot, 1)}


_ = LEVELS
