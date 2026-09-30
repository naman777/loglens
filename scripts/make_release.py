"""Assemble a Hugging Face-ready folder (release/loglens-v0.1) — does NOT upload anything.

Contents: FP32 + int8 ONNX, PyTorch weights, tokenizer(s), model card (with the metrics tables).
Upload later with `huggingface-cli upload <user>/loglens release/loglens-v0.1 .` after reviewing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release/loglens-v0.1"
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


def main(out: Path = OUT) -> None:
    missing = [src for src in [*FILES, "docs/model_card.md"]
               if not (ROOT / src).is_file() or (ROOT / src).stat().st_size == 0]
    if missing:
        raise SystemExit("Release incomplete; missing/empty files: " + ", ".join(missing))
    # A fresh destination prevents stale files or a previous manifest appearing valid.
    out.mkdir(parents=True, exist_ok=False)
    for src, dst in FILES.items():
        s = ROOT / src
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, out / dst)
    card = (ROOT / "docs/model_card.md").read_text(encoding="utf-8")
    card = card.replace("`results/benchmark.md`", "`docs/benchmark.md`")
    header = ("---\nlicense: mit\nlibrary_name: onnx\ntags: [logs, anomaly-detection, root-cause-analysis]\n"
              "---\n\n")
    (out / "README.md").write_text(header + card, encoding="utf-8")
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[path.relative_to(out).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.strip()
    (out / "manifest.json").write_text(json.dumps({"source_commit": commit,
        "source_dirty": bool(dirty), "files": files}, indent=2), encoding="utf-8")
    print("wrote", out, "-", sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6, "MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    main(parser.parse_args().out.resolve())
