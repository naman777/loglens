"""Export the line encoder and window model to ONNX (dynamic batch/sequence), verify parity against
PyTorch, and produce dynamic-int8 versions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from loglens.config import load_config, parse_args
from loglens.model.export_wrap import EncoderExport, WindowExport
from loglens.model.line_encoder import LineEncoder, LineEncoderConfig
from loglens.model.window_model import WindowModel, WindowModelConfig


@dataclass
class ExportConfig:
    encoder: str = "artifacts/pretrain/encoder.pt"
    window: str = "artifacts/window/sup/best.pt"
    tokenizer: str = "artifacts/tokenizer/loglens-bpe-16k.json"
    out_dir: str = "artifacts/onnx"
    opset: int = 17
    parity_n: int = 1000
    tol: float = 1e-3
    quantize: bool = True


def export_encoder(model: LineEncoder, path: Path, opset: int) -> None:
    model.eval()
    ids = torch.randint(5, model.cfg.vocab_size, (2, 12))
    ids[1, 8:] = 0
    torch.onnx.export(
        EncoderExport(model), (ids,), str(path), input_names=["ids"], output_names=["emb"],
        dynamic_axes={"ids": {0: "batch", 1: "seq"}, "emb": {0: "batch"}}, opset_version=opset,
        dynamo=False)
    model.eval()  # torch.onnx.export restores the wrapper's train() flag onto the shared submodules


def export_window(model: WindowModel, path: Path, opset: int) -> None:
    model.eval()
    b, L, d = 2, 10, model.cfg.in_dim
    args = (torch.randn(b, L, d), torch.randint(0, 16, (b, L)), torch.randint(0, 8, (b, L)),
            torch.randint(0, 8, (b, L)), torch.zeros(b, L, dtype=torch.bool))
    torch.onnx.export(
        WindowExport(model), args, str(path), input_names=["emb", "gap", "svc", "lvl", "pad"],
        output_names=["anomaly", "suspicion"],
        dynamic_axes={"emb": {0: "batch", 1: "seq"}, "gap": {0: "batch", 1: "seq"},
                      "svc": {0: "batch", 1: "seq"}, "lvl": {0: "batch", 1: "seq"},
                      "pad": {0: "batch", 1: "seq"}, "anomaly": {0: "batch"},
                      "suspicion": {0: "batch", 1: "seq"}}, opset_version=opset, dynamo=False)
    model.eval()


def quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def session(path: str | Path, threads: int | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def parity_encoder(model: LineEncoder, path: Path, n: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    sess = session(path)
    worst = 0.0
    for i in range(0, n, 100):
        L = int(rng.integers(6, 40))
        ids = rng.integers(5, model.cfg.vocab_size, (min(100, n - i), L)).astype(np.int64)
        ids[:, 0] = 1
        cut = rng.integers(3, L, len(ids))
        for r, c in enumerate(cut):
            ids[r, c:] = 0
        with torch.no_grad():
            ref = model(torch.from_numpy(ids)).numpy()
        out = sess.run(None, {"ids": ids})[0]
        worst = max(worst, float(np.abs(ref - out).max()))
    return worst


def parity_window(model: WindowModel, path: Path, n: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    sess = session(path)
    worst = 0.0
    for _ in range(max(n // 100, 1)):
        B, L = 8, int(rng.integers(8, 60))
        emb = rng.standard_normal((B, L, model.cfg.in_dim)).astype(np.float32)
        gap = rng.integers(0, 16, (B, L)).astype(np.int64)
        svc = rng.integers(0, 8, (B, L)).astype(np.int64)
        lvl = rng.integers(0, 8, (B, L)).astype(np.int64)
        pad = np.zeros((B, L), dtype=bool)
        pad[:, L - 3:] = True
        with torch.no_grad():
            o = model(*(torch.from_numpy(x) for x in (emb, gap, svc, lvl, pad)))
        a, s = sess.run(None, {"emb": emb, "gap": gap, "svc": svc, "lvl": lvl, "pad": pad})
        valid = ~pad
        worst = max(worst, float(np.abs(o["anomaly"].numpy() - a).max()),
                    float(np.abs((o["suspicion"].numpy() - s)[valid]).max()))
    return worst


def main(cfg: ExportConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(cfg.encoder, map_location="cpu", weights_only=False)
    enc = LineEncoder(LineEncoderConfig(**ck["cfg"]))
    enc.load_state_dict(ck["model"])
    enc.eval()
    wk = torch.load(cfg.window, map_location="cpu", weights_only=False)
    from loglens.train.finetune import FinetuneConfig, make_model

    win = make_model(FinetuneConfig(**wk["cfg"]))
    win.load_state_dict(wk["model"])
    win.eval()
    export_encoder(enc, out / "encoder.onnx", cfg.opset)
    export_window(win, out / "window.onnx", cfg.opset)
    report = {"encoder_max_abs_diff": parity_encoder(enc, out / "encoder.onnx", cfg.parity_n),
              "window_max_abs_diff": parity_window(win, out / "window.onnx", cfg.parity_n)}
    assert report["encoder_max_abs_diff"] < cfg.tol, report
    assert report["window_max_abs_diff"] < cfg.tol, report
    if cfg.quantize:
        quantize(out / "encoder.onnx", out / "encoder.int8.onnx")
        quantize(out / "window.onnx", out / "window.int8.onnx")
    for f in out.glob("*.onnx"):
        report[f"{f.name}_mb"] = round(f.stat().st_size / 1e6, 2)
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    _ = WindowModelConfig
    return report


if __name__ == "__main__":
    args = parse_args("onnx export")
    main(load_config(ExportConfig, args.config))
