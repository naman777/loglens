"""Accuracy of the ONNX models (FP32 vs dynamic int8) on the shared windows: F1 / PR-AUC at a
validation-tuned threshold plus lab RCA Recall@5, all through onnxruntime on CPU."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from loglens.config import load_config, parse_args
from loglens.eval.metrics import evaluate_scores, rank_of_first_causal, rca_metrics
from loglens.serve.export_onnx import session
from loglens.train.windata import SystemData


@dataclass
class Int8Config:
    onnx_dir: str = "artifacts/onnx"
    tok_dir: str = "data/tok"
    win_dir: str = "data/win"
    systems: list[str] = field(default_factory=lambda: ["BGL", "HDFS"])
    max_windows: int = 6000
    threads: int = 4
    out_json: str = "results/int8_accuracy.json"
    seed: int = 1337


def ort_embed(sess, ids: np.ndarray, offsets: np.ndarray, max_len: int = 128, bs: int = 256) -> np.ndarray:
    n = len(offsets) - 1
    lens = np.minimum(np.diff(offsets), max_len)
    order = np.argsort(lens, kind="stable")
    out = np.zeros((n, 256), dtype=np.float32)
    for i in range(0, n, bs):
        rows = order[i: i + bs]
        L = int(lens[rows].max())
        b = np.zeros((len(rows), L), dtype=np.int64)
        for r, k in enumerate(rows):
            b[r, : lens[k]] = ids[offsets[k]: offsets[k] + lens[k]]
        out[rows] = sess.run(None, {"ids": b})[0]
    return out


def run_windows(sess, sd: SystemData, emb: np.ndarray, widx: np.ndarray, bs: int = 32) -> np.ndarray:
    out = np.zeros(len(widx))
    for i in range(0, len(widx), bs):
        spans = [sd.span(int(w)) for w in widx[i: i + bs]]
        L = max(b - a for a, b in spans)
        B = len(spans)
        feed = {"emb": np.zeros((B, L, 256), np.float32), "gap": np.zeros((B, L), np.int64),
                "svc": np.zeros((B, L), np.int64), "lvl": np.zeros((B, L), np.int64),
                "pad": np.ones((B, L), bool)}
        for r, (a, b) in enumerate(spans):
            n = b - a
            feed["emb"][r, :n] = emb[sd.z["uid"][a:b]]
            feed["gap"][r, :n] = sd.z["gap"][a:b]
            feed["svc"][r, :n] = sd.z["svc"][a:b]
            feed["lvl"][r, :n] = sd.z["lvl"][a:b]
            feed["pad"][r, :n] = False
        out[i: i + B] = sess.run(None, feed)[0]
    return out


def main(cfg: Int8Config) -> dict:
    res: dict = {}
    for tag, suffix in (("fp32", ".onnx"), ("int8", ".int8.onnx")):
        enc = session(Path(cfg.onnx_dir) / f"encoder{suffix}", cfg.threads)
        win = session(Path(cfg.onnx_dir) / f"window{suffix}", cfg.threads)
        res[tag] = {}
        for s in cfg.systems:
            sd = SystemData(s, cfg.win_dir, "no_emb")
            z = np.load(Path(cfg.tok_dir) / f"{s}.npz")
            emb = ort_embed(enc, z["ids"], z["offsets"])
            rng = np.random.default_rng(cfg.seed)

            def pick(sp):
                w = sd.windows(sp)
                return np.sort(rng.choice(w, cfg.max_windows, replace=False)) if len(w) > cfg.max_windows else w

            wv, wt = pick("val"), pick("test")
            yv, yt = sd.z["wlabel"][wv], sd.z["wlabel"][wt]
            r = evaluate_scores(yv, run_windows(win, sd, emb, wv), yt, run_windows(win, sd, emb, wt))
            if s == "Lab" and sd.incidents:
                ranks = []
                for k, (_iid, _ft, _tg, sp) in enumerate(sd.incidents):
                    if sp != "test":
                        continue
                    a, b = int(sd.z["inc_a"][k]), int(sd.z["inc_b"][k])
                    score = np.full(b - a, -1e9)
                    st = a
                    while True:
                        e = min(st + 256, b)
                        spans_emb = emb[sd.z["uid"][st:e]][None]
                        feed = {"emb": spans_emb, "gap": sd.z["gap"][st:e][None].astype(np.int64),
                                "svc": sd.z["svc"][st:e][None].astype(np.int64),
                                "lvl": sd.z["lvl"][st:e][None].astype(np.int64),
                                "pad": np.zeros((1, e - st), bool)}
                        sl = win.run(None, feed)[1][0]
                        score[st - a: e - a] = np.maximum(score[st - a: e - a], sl)
                        if e >= b:
                            break
                        st += 128
                    ranks.append(rank_of_first_causal(score, sd.z["causal"][a:b]))
                r["rca"] = rca_metrics(ranks)
            res[tag][s] = r
            print(tag, s, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
    res["f1_drop"] = {s: res["fp32"][s]["f1"] - res["int8"][s]["f1"] for s in cfg.systems}
    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_json).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res["f1_drop"], indent=2))
    return res


if __name__ == "__main__":
    args = parse_args("int8 accuracy")
    main(load_config(Int8Config, args.config))
