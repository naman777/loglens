"""Metric definitions (Implementation Plan, Appendix B). Window-level P/R/F1 at a validation-tuned
threshold, PR-AUC, RCA Recall@k / MRR, and throughput helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = float((y_true & y_pred).sum())
    fp = float((~y_true & y_pred).sum())
    fn = float((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, scores))


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 -- call on VALIDATION data only, then freeze."""
    y = np.asarray(y_true)
    if y.sum() == 0 or y.sum() == len(y):
        return float(np.median(scores))
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[int(np.argmax(f1[:-1]))]) if len(t) else 0.5


def evaluate_scores(y_val, s_val, y_test, s_test) -> dict[str, float]:
    thr = best_threshold(y_val, s_val)
    out = prf(y_test, np.asarray(s_test) >= thr)
    out.update({"pr_auc": pr_auc(y_test, s_test), "threshold": thr})
    return out


def rank_of_first_causal(scores: np.ndarray, causal: np.ndarray) -> int | None:
    """1-based rank of the best-ranked causal line (ties broken pessimistically)."""
    causal = np.asarray(causal).astype(bool)
    if not causal.any():
        return None
    best = np.max(np.asarray(scores)[causal])
    return int((np.asarray(scores) > best).sum() + 1 + ((np.asarray(scores) == best).sum() - 1))


def rca_metrics(ranks: list[int | None], ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    valid = [r for r in ranks if r is not None]
    n = max(len(valid), 1)
    out = {f"recall@{k}": sum(r <= k for r in valid) / n for k in ks}
    out["mrr"] = sum(1 / r for r in valid) / n
    out["n_incidents"] = len(valid)
    return out


def percentile_ms(latencies_s: list[float]) -> dict[str, float]:
    a = np.asarray(latencies_s) * 1000
    return {"p50_ms": float(np.percentile(a, 50)), "p99_ms": float(np.percentile(a, 99))}
