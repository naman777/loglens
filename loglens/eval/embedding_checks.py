"""Phase 4.6 sanity checks on the line embeddings -> docs/embedding_report.md (+ t-SNE PNGs).

1. Nearest neighbours: for random lines, do the 5 nearest neighbours share the query's Drain3
   template? (automatic judge instead of eyeballing; examples are printed for a human look).
2. t-SNE coloured by system and by level (the plan says UMAP; sklearn t-SNE is used, no extra dep).
3. Linear probe on BGL line labels: frozen encoder + logistic regression vs TF-IDF + logistic
   regression, trained on train-split lines and tested on lines whose masked text first appears
   *after* the train split (no template lookup possible).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from loglens.config import load_config, parse_args
from loglens.seed import set_seed


@dataclass
class EmbCheckConfig:
    seed: int = 1337
    masked_dir: str = "data/masked"
    emb_dir: str = "data/emb"
    out_md: str = "docs/embedding_report.md"
    fig_dir: str = "docs/figures"
    n_queries: int = 200
    per_system: int = 800
    systems: list[str] = field(default_factory=lambda: [
        "BGL", "HDFS", "Lab", "OpenStack", "Hadoop", "Spark", "Zookeeper", "Linux", "Apache", "HPC",
        "SSH", "Mac", "HealthApp", "Proxifier", "Android", "Thunderbird"])
    probe_system: str = "BGL"


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def load(cfg: EmbCheckConfig):
    rows = []
    for s in cfg.systems:
        u = pl.read_parquet(Path(cfg.masked_dir) / s / "uniques.parquet").sort("uid")
        e = np.load(Path(cfg.emb_dir) / f"{s}.npy")
        n = min(cfg.per_system, len(u))
        idx = np.random.default_rng(cfg.seed).choice(len(u), n, replace=False)
        rows.append((s, u[idx.tolist()], e[idx]))
    return rows


def nn_check(cfg: EmbCheckConfig, rows) -> tuple[dict, list[str]]:
    from loglens.eval.baselines.classic import drain_clusters

    rng = np.random.default_rng(cfg.seed)
    top1, top5, n = 0, 0.0, 0
    examples: list[str] = []
    for s, u, e in rows:
        if s == "Thunderbird" or len(u) < 20:
            continue
        clusters_all = drain_clusters(s, cfg.masked_dir, 120_000)
        cl = clusters_all[u["uid"].to_numpy()]
        z = _norm(e)
        q = rng.choice(len(u), min(cfg.n_queries // 10, len(u)), replace=False)
        sim = z[q] @ z.T
        sim[np.arange(len(q)), q] = -1
        nn = np.argsort(-sim, axis=1)[:, :5]
        for qi, row in zip(q, nn, strict=True):
            same = cl[row] == cl[qi]
            top1 += bool(same[0])
            top5 += float(same.mean())
            n += 1
            if len(examples) < 6:
                examples.append(f"- **{s}**: `{u['masked'][int(qi)][:90]}`\n  - NN1: `{u['masked'][int(row[0])][:90]}`")
    return {"top1_same_template": top1 / max(n, 1), "top5_same_template": top5 / max(n, 1), "n": n}, examples


def tsne_plots(cfg: EmbCheckConfig, rows) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    X = np.concatenate([_norm(e[:400]) for _, _, e in rows])
    sysl = np.concatenate([[s] * min(400, len(e)) for s, _, e in rows])
    lvl = np.concatenate([u["level"].to_numpy()[:400] for _, u, _ in rows])
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=cfg.seed).fit_transform(X)
    Path(cfg.fig_dir).mkdir(parents=True, exist_ok=True)
    out = []
    for name, labels in (("by_system", sysl), ("by_level", lvl)):
        fig, ax = plt.subplots(figsize=(8, 6))
        for lab in sorted(set(labels)):
            m = labels == lab
            ax.scatter(xy[m, 0], xy[m, 1], s=4, label=str(lab))
        ax.legend(markerscale=3, fontsize=7, ncol=2)
        ax.set_title(f"Line embeddings (t-SNE) {name}")
        p = Path(cfg.fig_dir) / f"tsne_{name}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        out.append(str(p))
    return out


def linear_probe(cfg: EmbCheckConfig) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score

    s = cfg.probe_system
    u = pl.read_parquet(Path(cfg.masked_dir) / s / "uniques.parquet").sort("uid")
    lines = pl.read_parquet(Path(cfg.masked_dir) / s / "lines.parquet", columns=["uid", "label"])
    lab = lines.group_by("uid").agg(pl.col("label").mean().alias("rate"))
    u = u.join(lab, on="uid", how="left").sort("uid")
    emb = np.load(Path(cfg.emb_dir) / f"{s}.npy").astype(np.float32)
    y = (u["rate"].fill_null(0).to_numpy() > 0.5).astype(int)
    tr = u["count_train"].to_numpy() > 0
    te = ~tr
    res = {"n_train": int(tr.sum()), "n_test_novel": int(te.sum()),
           "test_pos_rate": float(y[te].mean()) if te.sum() else float("nan")}
    if te.sum() < 20 or y[te].sum() == 0 or y[tr].sum() == 0:
        return res | {"note": "too few novel positives for a probe"}
    for name, Xtr, Xte in (
        ("encoder", emb[tr], emb[te]),
        ("tfidf", *(lambda v: (v.fit_transform(u["masked"].to_numpy()[tr]),
                               v.transform(u["masked"].to_numpy()[te])))(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True))),
    ):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        res[name] = {"pr_auc": float(average_precision_score(y[te], p)),
                     "f1": float(f1_score(y[te], p > 0.5))}
    return res


def main(cfg: EmbCheckConfig) -> None:
    set_seed(cfg.seed)
    rows = load(cfg)
    nn, ex = nn_check(cfg, rows)
    figs = tsne_plots(cfg, rows)
    probe = linear_probe(cfg)
    md = ["# Embedding sanity report", "",
          "## Nearest neighbours", "",
          f"Automatic judge on {nn['n']} random lines: the nearest neighbour shares the query's "
          f"Drain3 template in **{nn['top1_same_template'] * 100:.1f}%** of cases; mean fraction of "
          f"the 5 nearest neighbours sharing it: {nn['top5_same_template'] * 100:.1f}%. "
          "(Gate: >= 90% of spot checks. Lines here are *distinct masked strings*, so a neighbour "
          "with a different string but the same Drain3 template is the meaningful signal.)", "",
          "Examples:", "", *ex, "",
          "## t-SNE", "", *(f"![{Path(f).stem}](figures/{Path(f).name})" for f in figs), "",
          f"## Linear probe on {cfg.probe_system} line labels", "",
          "Trained on lines seen in the train split, tested on lines whose masked text first "
          "appears later (template lookup impossible).", "",
          "```json", __import__("json").dumps(probe, indent=2), "```", ""]
    Path(cfg.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_md).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    args = parse_args("embedding sanity checks")
    main(load_config(EmbCheckConfig, args.config))
