"""Assemble a Hugging Face-ready folder (release/loglens-v0.1) — does NOT upload anything.

Contents: FP32 + int8 ONNX, PyTorch weights, tokenizer(s), model card (with the metrics tables).
Upload later with `huggingface-cli upload <user>/loglens release/loglens-v0.1 .` after reviewing.
"""
from __future__ import annotations

import shutil
from pathlib import Path

OUT = Path("release/loglens-v0.1")
FILES = {
    "artifacts/onnx/encoder.onnx": "onnx/encoder.onnx",
    "artifacts/onnx/encoder.int8.onnx": "onnx/encoder.int8.onnx",
    "artifacts/onnx/window.onnx": "onnx/window.onnx",
    "artifacts/onnx/window.int8.onnx": "onnx/window.int8.onnx",
    "artifacts/pretrain/encoder.pt": "pytorch/line_encoder.pt",
    "artifacts/window/sup/best.pt": "pytorch/window_model_supervised.pt",
    "artifacts/window/unsup/best.pt": "pytorch/window_model_unsupervised.pt",
    "artifacts/tokenizer/loglens-bpe-16k.json": "tokenizer/loglens-bpe-16k.json",
    "artifacts/tokenizer/loglens-bpe-8k.json": "tokenizer/loglens-bpe-8k.json",
    "docs/tokenizer_report.md": "docs/tokenizer_report.md",
    "results/benchmark.md": "docs/benchmark.md",
    "results/ablations.md": "docs/ablations.md",
    "docs/data_report.md": "docs/data_report.md",
    "docs/serving_benchmark.md": "docs/serving_benchmark.md",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src, dst in FILES.items():
        s = Path(src)
        if not s.exists():
            print("missing", src)
            continue
        (OUT / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, OUT / dst)
    card = Path("docs/model_card.md").read_text(encoding="utf-8")
    header = ("---\nlicense: mit\nlibrary_name: onnx\ntags: [logs, anomaly-detection, root-cause-analysis]\n"
              "---\n\n")
    (OUT / "README.md").write_text(header + card, encoding="utf-8")
    print("wrote", OUT, "-", sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6, "MB")


if __name__ == "__main__":
    main()
