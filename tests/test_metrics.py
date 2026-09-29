import numpy as np

from loglens.eval.metrics import (
    best_threshold, evaluate_scores, prf, rank_of_first_causal, rca_metrics,
)


def test_prf_basic():
    m = prf(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert m == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_rank_and_rca():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    causal = np.array([0, 0, 1, 0])
    assert rank_of_first_causal(scores, causal) == 3
    assert rank_of_first_causal(scores, np.zeros(4)) is None
    m = rca_metrics([1, 3, 10, None])
    assert m["recall@1"] == 1 / 3 and m["recall@5"] == 2 / 3
    assert abs(m["mrr"] - (1 + 1 / 3 + 0.1) / 3) < 1e-9


def test_ties_are_pessimistic():
    assert rank_of_first_causal(np.array([1.0, 1.0, 1.0]), np.array([0, 1, 0])) == 3


def test_threshold_tuned_on_val_only():
    rng = np.random.default_rng(0)
    yv = np.array([0] * 90 + [1] * 10)
    sv = np.concatenate([rng.random(90) * 0.5, 0.6 + rng.random(10) * 0.4])
    thr = best_threshold(yv, sv)
    assert 0.4 < thr < 0.7
    res = evaluate_scores(yv, sv, yv, sv)
    assert res["f1"] > 0.9 and res["pr_auc"] > 0.95
