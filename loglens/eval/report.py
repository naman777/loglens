"""`make eval`: run every method on the shared windows/splits and write results/results.json and
results/benchmark.md. All numbers come from this one command with a fixed seed.

Experiments
  E1 in-distribution anomaly detection (BGL, HDFS; time split)
  E2 unseen-system transfer (Thunderbird): baselines train on it, LogLens does not (zero-shot),
     plus LogLens fine-tuned with 1% of its train labels
  E3 root-cause ranking on lab test incidents (Recall@1/@5, MRR, per fault type)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from loglens.config import load_config, parse_args
from loglens.eval.baselines.classic import (
    DeepLogBaseline, TemplateCountIForest, drain_clusters, random_rank, severity_rank,
)
from loglens.eval.metrics import evaluate_scores, prf, rank_of_first_causal, rca_metrics
from loglens.model.window_model import WindowModel, WindowModelConfig
from loglens.seed import set_seed
from loglens.tokenizer.masking import LEVELS
from loglens.train.finetune import (
    FinetuneConfig, rank_incident, sup_scores, train, unsup_scores,
)
from loglens.train.windata import SystemData


@dataclass
class EvalConfig:
    seed: int = 1337
    win_dir: str = "data/win"
    emb_dir: str = "data/emb"
    masked_dir: str = "data/masked"
    out_dir: str = "results"
    sup_ckpt: str = "artifacts/window/sup/best.pt"
    unsup_ckpt: str = "artifacts/window/unsup/best.pt"
    anomaly_systems: list[str] = field(default_factory=lambda: ["BGL", "HDFS"])
    unseen: str = "Thunderbird"
    max_eval_windows: int = 40_000
    few_label_frac: float = 0.01
    few_label_steps: int = 300
    unseen_frac_seeds: int = 3
    dev: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_window_model(path: str, dev: str) -> tuple[WindowModel, FinetuneConfig]:
    st = torch.load(path, map_location=dev, weights_only=False)
    fc = FinetuneConfig(**st["cfg"])
    from loglens.train.finetune import make_model

    m = make_model(fc).to(dev)
    m.load_state_dict(st["model"])
    return m.eval(), fc


def subsample(idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(idx) <= n:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


def anomaly_table(cfg: EvalConfig, system: str, sup, unsup, fc_sup, fc_unsup, extra=None) -> dict:
    sd = SystemData(system, cfg.win_dir, cfg.emb_dir)
    wv = subsample(sd.windows("val"), cfg.max_eval_windows, cfg.seed)
    wt = subsample(sd.windows("test"), cfg.max_eval_windows, cfg.seed + 1)
    yv, yt = sd.z["wlabel"][wv], sd.z["wlabel"][wt]
    res: dict = {"_n_val": int(len(wv)), "_n_test": int(len(wt)),
                 "_test_positive_rate": float(yt.mean())}
    res["always-anomalous"] = {**prf(yt, np.ones_like(yt)), "pr_auc": float(yt.mean())}
    clusters = drain_clusters(system, cfg.masked_dir)
    t0 = time.time()
    iso = TemplateCountIForest(sd, clusters, seed=cfg.seed).fit()
    res["Drain3+IsolationForest"] = evaluate_scores(yv, iso.score_windows(wv), yt, iso.score_windows(wt))
    dl = DeepLogBaseline(sd, clusters, seed=cfg.seed, dev=cfg.dev).fit()
    res["Drain3+DeepLog"] = evaluate_scores(yv, dl.score_windows(wv), yt, dl.score_windows(wt))
    res["_baseline_secs"] = time.time() - t0
    if unsup is not None:
        res["LogLens-unsup"] = evaluate_scores(
            yv, unsup_scores(unsup, sd, wv, fc_unsup, cfg.dev), yt, unsup_scores(unsup, sd, wt, fc_unsup, cfg.dev))
    if sup is not None:
        res["LogLens-sup"] = evaluate_scores(
            yv, sup_scores(sup, sd, wv, fc_sup, cfg.dev), yt, sup_scores(sup, sd, wt, fc_sup, cfg.dev))
    return res


def unseen_few_label(cfg: EvalConfig, fc_sup: FinetuneConfig, sup_path: str) -> dict:
    """Fine-tune the supervised LogLens on ``few_label_frac`` of the unseen system's train windows."""
    sd = SystemData(cfg.unseen, cfg.win_dir, cfg.emb_dir)
    wv = subsample(sd.windows("val"), cfg.max_eval_windows, cfg.seed)
    wt = subsample(sd.windows("test"), cfg.max_eval_windows, cfg.seed + 1)
    yv, yt = sd.z["wlabel"][wv], sd.z["wlabel"][wt]
    out = []
    for k in range(cfg.unseen_frac_seeds):
        from dataclasses import replace

        fc = replace(fc_sup, seed=cfg.seed + k, init=sup_path, train_systems=[cfg.unseen],
                     eval_systems=[cfg.unseen], steps=cfg.few_label_steps, eval_every=50, patience=4,
                     run_name=f"fewlabel_{k}", out_dir=f"artifacts/window/fewlabel_{k}", lr=1e-4,
                     warmup=20)
        # restrict train windows to a random 1% subset via a temporary index override
        import loglens.train.windata as wd

        full = wd.SystemData.windows

        def limited(self, split, labeled="any", _k=k, _full=full):
            w = _full(self, split, labeled)
            if split == "train" and self.system == cfg.unseen:
                n = max(int(len(w) * cfg.few_label_frac), 8)
                return np.sort(np.random.default_rng(cfg.seed + _k).choice(w, n, replace=False))
            return w

        wd.SystemData.windows = limited
        try:
            train(fc)
        finally:
            wd.SystemData.windows = full
        m, fcx = load_window_model(f"artifacts/window/fewlabel_{k}/best.pt", cfg.dev)
        out.append(evaluate_scores(yv, sup_scores(m, sd, wv, fcx, cfg.dev),
                                   yt, sup_scores(m, sd, wt, fcx, cfg.dev)))
    keys = [k for k in out[0] if k != "threshold"]
    return {k: float(np.mean([o[k] for o in out])) for k in keys} | {
        "f1_std": float(np.std([o["f1"] for o in out])), "n_seeds": len(out),
        "train_windows": max(int(len(sd.windows('train')) * cfg.few_label_frac), 8)}


def rca_table(cfg: EvalConfig, sup, fc_sup, extra_methods: dict | None = None) -> dict:
    sd = SystemData("Lab", cfg.win_dir, cfg.emb_dir)
    clusters = drain_clusters("Lab", cfg.masked_dir)
    dl = DeepLogBaseline(sd, clusters, seed=cfg.seed, dev=cfg.dev).fit()
    levels = LEVELS
    methods = {
        "random": lambda a, b, k: random_rank(a, b, seed=cfg.seed + k),
        "severity-heuristic": lambda a, b, k: severity_rank(sd, a, b, levels),
        "Drain3+DeepLog (surprise)": lambda a, b, k: dl.rank_lines(a, b),
    }
    if sup is not None:
        methods["LogLens-sup"] = lambda a, b, k: rank_incident(sup, sd, a, b, fc_sup, cfg.dev)
    if extra_methods:
        methods.update(extra_methods)
    out = {}
    for name, fn in methods.items():
        ranks, by_type = [], {}
        for k, (iid, ftype, tgt, sp) in enumerate(sd.incidents):
            if sp != "test":
                continue
            a, b = int(sd.z["inc_a"][k]), int(sd.z["inc_b"][k])
            r = rank_of_first_causal(fn(a, b, k), sd.z["causal"][a:b])
            ranks.append(r)
            by_type.setdefault(ftype, []).append(r)
        out[name] = {**rca_metrics(ranks), "by_fault_type": {t: rca_metrics(v) for t, v in by_type.items()}}
    return out


def fmt(x, p=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def render(res: dict) -> str:
    md = ["# Benchmark", "", "Fixed seed; all numbers produced by `make eval`. Anomaly metrics are "
          "window-level at a threshold tuned on the validation split only (test never touched). "
          "Lab incidents are **synthetic** (simulated microservice lab).", ""]
    for sysname, r in res["anomaly"].items():
        tag = " (unseen system)" if sysname == res["unseen"] else ""
        md += [f"## Anomaly detection: {sysname}{tag}", "",
               f"Test windows: {r['_n_test']:,} ({r['_test_positive_rate'] * 100:.1f}% positive).", "",
               "| method | precision | recall | F1 | PR-AUC |", "| --- | --- | --- | --- | --- |"]
        for m, v in r.items():
            if m.startswith("_") or not isinstance(v, dict):
                continue
            md.append(f"| {m} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['f1'])} | {fmt(v['pr_auc'])} |")
        md.append("")
    if "few_label" in res:
        v = res["few_label"]
        md += [f"LogLens-sup fine-tuned on {v['train_windows']} labelled {res['unseen']} windows "
               f"(1%): F1 {fmt(v['f1'])} ± {fmt(v['f1_std'])}, PR-AUC {fmt(v['pr_auc'])} "
               f"({v['n_seeds']} seeds).", ""]
    md += ["## Root-cause ranking (lab test incidents)", "",
           "| method | Recall@1 | Recall@5 | MRR | incidents |", "| --- | --- | --- | --- | --- |"]
    for m, v in res["rca"].items():
        md.append(f"| {m} | {fmt(v['recall@1'])} | {fmt(v['recall@5'])} | {fmt(v['mrr'])} | {v['n_incidents']} |")
    md += ["", "Per fault type, Recall@5:", ""]
    types = sorted({t for v in res["rca"].values() for t in v["by_fault_type"]})
    md += ["| method | " + " | ".join(types) + " |", "| --- |" + " --- |" * len(types)]
    for m, v in res["rca"].items():
        md.append(f"| {m} | " + " | ".join(
            fmt(v["by_fault_type"].get(t, {}).get("recall@5")) for t in types) + " |")
    return "\n".join(md) + "\n"


def main(cfg: EvalConfig) -> dict:
    set_seed(cfg.seed)
    sup = fc_sup = unsup = fc_unsup = None
    if Path(cfg.sup_ckpt).exists():
        sup, fc_sup = load_window_model(cfg.sup_ckpt, cfg.dev)
    if Path(cfg.unsup_ckpt).exists():
        unsup, fc_unsup = load_window_model(cfg.unsup_ckpt, cfg.dev)
    res: dict = {"unseen": cfg.unseen, "anomaly": {}}
    for s in [*cfg.anomaly_systems, cfg.unseen]:
        print("anomaly", s, flush=True)
        res["anomaly"][s] = anomaly_table(cfg, s, sup, unsup, fc_sup, fc_unsup)
    if sup is not None:
        print("few-label", flush=True)
        res["few_label"] = unseen_few_label(cfg, fc_sup, cfg.sup_ckpt)
    print("rca", flush=True)
    res["rca"] = rca_table(cfg, sup, fc_sup)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "benchmark.md").write_text(render(res), encoding="utf-8")
    print(render(res))
    _ = WindowModelConfig
    return res


if __name__ == "__main__":
    args = parse_args("run all evaluations")
    main(load_config(EvalConfig, args.config))
