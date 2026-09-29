import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from loglens.model.line_encoder import LineEncoder, LineEncoderConfig
from loglens.model.window_model import WindowModel, WindowModelConfig
from loglens.serve.export_onnx import export_encoder, export_window, quantize
from loglens.serve.runtime import LogLensRuntime, RuntimeConfig
from loglens.tokenizer.tok import DEFAULT_SERVICES
from loglens.tokenizer.train_bpe import train_one


def _lines(n=300):
    out = []
    for i in range(n):
        out.append(f'{{"ts": {1772323200000 + i * 40}, "service": "orders", "level": "INFO", '
                   f'"message": "db ok: INSERT order {100000 + i} in {i % 9}ms pool_in_use=3/10"}}')
    out[150] = '{"ts": 1772323206000, "service": "payments", "level": "ERROR", "message": "bank timeout after 5000ms"}'
    return out


def test_runtime_end_to_end(tmp_path):
    texts = [f"<LVL:INFO> <SVC:orders> db ok: INSERT order <NUM> in <DURATION:<10ms> pool_in_use=<NUM>/<NUM> {i}"
             for i in range(200)] + ["<LVL:ERROR> <SVC:payments> bank timeout after <DURATION:10s+>"] * 20
    tk = train_one(texts, 400, list(DEFAULT_SERVICES), tmp_path / "tok.json")
    enc = LineEncoder(LineEncoderConfig(vocab_size=tk.get_vocab_size(), d_model=32, n_layers=2,
                                        n_heads=4, d_ff=64, embed_dim=256, max_len=64)).eval()
    win = WindowModel(WindowModelConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, max_window=256,
                                        n_services=32)).eval()
    export_encoder(enc, tmp_path / "encoder.onnx", 17)
    export_window(win, tmp_path / "window.onnx", 17)
    quantize(tmp_path / "encoder.onnx", tmp_path / "encoder.int8.onnx")
    quantize(tmp_path / "window.onnx", tmp_path / "window.int8.onnx")
    lines = _lines()
    res = {}
    for int8 in (False, True):
        rt = LogLensRuntime(RuntimeConfig(model_dir=str(tmp_path), tokenizer=str(tmp_path / "tok.json"),
                                          int8=int8, threads=1))
        r = rt.score(lines, top_k=5)
        assert r.n_lines == len(lines) and len(r.lines) == 5 and 0 <= r.anomaly <= 1
        first_hits = rt.cache.hit_rate
        rt.score(lines, top_k=5)
        assert rt.cache.hit_rate > first_hits  # second pass is served from the cache
        assert rt.metrics()["lines"] == 2 * len(lines)
        res[int8] = np.array([s["score"] for s in r.lines])
    assert res[False].shape == res[True].shape
