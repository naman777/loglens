"""Train and test the ablation variants of the supervised window model.

Variants: full (reference), no time-gap embedding, randomly-initialised line encoder, and (if
available) the SimCSE-pretrained encoder. Every variant uses the same splits/seed/steps; thresholds
come from validation only. Writes results/ablations.json and results/ablations.md.

usage: python scripts/run_ablations.py [--variants full,nogap,random_enc,...]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loglens.eval.metrics import evaluate_scores  # noqa: E402
from loglens.seed import get_device  # noqa: E402
from loglens.train.finetune import FinetuneConfig, rca_eval, sup_scores, train  # noqa: E402
from loglens.train.windata import SystemData  # noqa: E402

BASE = FinetuneConfig(
    mode="sup", train_systems=["BGL", "HDFS", "Lab"], eval_systems=["BGL", "HDFS", "Lab"], steps=5000,
    eval_every=250, patience=8, batch_size=32, sampler_power=0.3, run_name="ablation")

TRANSFER = ["Thunderbird"]  # unseen system: zero-shot test metrics only

VARIANTS = {
    "full": {},
    "nogap": {"use_gap": False},
    "random_enc": {"emb_dir": "data/emb_random", "pretrained_encoder": False},
    "simcse_enc": {"emb_dir": "data/emb_v2"},
    # masking ablation: durations collapsed to <NUM> (LOGLENS_MASK_DURATIONS=0), random-init encoder,
    # compared against random_enc (bucketed durations, same encoder type)
    "random_enc_nodur": {"emb_dir": "data/emb_nodur_random", "win_dir": "data/win_nodur",
                         "pretrained_encoder": False},
    "template_enc": {"emb_dir": "data/emb_v3"},
}


def test_metrics(cfg: FinetuneConfig, ckpt: str, dev: str) -> dict:
    import torch

    from loglens.train.finetune import make_model

    st = torch.load(ckpt, map_location=dev, weights_only=False)
    model = make_model(FinetuneConfig(**st["cfg"])).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    out = {}
    for s in [*cfg.eval_systems, *TRANSFER]:
        sd = SystemData(s, cfg.win_dir, cfg.emb_dir)
        rng = np.random.default_rng(cfg.seed)

        def pick(sp):
            w = sd.windows(sp)
            return np.sort(rng.choice(w, 20000, replace=False)) if len(w) > 20000 else w

        wv, wt = pick("val"), pick("test")
        out[s] = evaluate_scores(sd.z["wlabel"][wv], sup_scores(model, sd, wv, cfg, dev),
                                 sd.z["wlabel"][wt], sup_scores(model, sd, wt, cfg, dev))
        if sd.incidents:
            r = rca_eval(model, sd, "test", cfg, dev)
            out[s]["rca_recall@5"] = r["recall@5"]
            out[s]["rca_recall@1"] = r["recall@1"]
            out[s]["rca_mrr"] = r["mrr"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="full,nogap,random_enc")
    ap.add_argument("--eval-only", action="store_true", help="reuse trained checkpoints")
    a = ap.parse_args()
    dev = get_device()
    res = {}
    f = Path("results/ablations.json")
    if f.exists():
        res = json.loads(f.read_text())
    for v in a.variants.split(","):
        cfg = replace(BASE, out_dir=f"artifacts/window/abl_{v}", run_name=f"abl_{v}", **VARIANTS[v])
        print("== training" if not a.eval_only else "== evaluating", v, flush=True)
        r = {"out": f"artifacts/window/abl_{v}/best.pt"} if a.eval_only else train(cfg)
        res[v] = test_metrics(cfg, r["out"], dev)
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps(res, indent=2, default=float))
    md = ["# Ablations (test split, supervised window model)", "",
          "| variant | BGL F1 | BGL PR-AUC | HDFS F1 | HDFS PR-AUC | Lab F1 | Lab PR-AUC | RCA R@1 | RCA R@5 | MRR | "
          "Thunderbird zero-shot F1 | Thunderbird zero-shot PR-AUC |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v, r in res.items():
        md.append(f"| {v} | {r['BGL']['f1']:.3f} | {r['BGL']['pr_auc']:.3f} | {r['HDFS']['f1']:.3f} | "
                  f"{r['HDFS']['pr_auc']:.3f} | {r['Lab']['f1']:.3f} | {r['Lab']['pr_auc']:.3f} | "
                  f"{r['Lab']['rca_recall@1']:.3f} | {r['Lab']['rca_recall@5']:.3f} | {r['Lab']['rca_mrr']:.3f} | "
                  f"{r['Thunderbird']['f1']:.3f} | {r['Thunderbird']['pr_auc']:.3f} |")
    Path("results/ablations.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
