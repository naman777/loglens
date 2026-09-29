import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from loglens.model.line_encoder import LineEncoder, LineEncoderConfig
from loglens.model.window_model import WindowModel, WindowModelConfig
from loglens.serve.export_onnx import (
    export_encoder,
    export_window,
    parity_encoder,
    parity_window,
    quantize,
)


def test_onnx_parity_and_int8(tmp_path):
    enc = LineEncoder(LineEncoderConfig(vocab_size=200, d_model=32, n_layers=2, n_heads=4,
                                        d_ff=64, embed_dim=16, max_len=64, dropout=0.1)).eval()
    export_encoder(enc, tmp_path / "e.onnx", 17)
    assert parity_encoder(enc, tmp_path / "e.onnx", 200) < 1e-3
    win = WindowModel(WindowModelConfig(in_dim=16, d_model=32, n_layers=2, n_heads=4, d_ff=64,
                                        max_window=64, n_services=8, n_levels=8)).eval()
    export_window(win, tmp_path / "w.onnx", 17)
    assert parity_window(win, tmp_path / "w.onnx", 200) < 1e-3
    quantize(tmp_path / "e.onnx", tmp_path / "e8.onnx")
    assert (tmp_path / "e8.onnx").stat().st_size < (tmp_path / "e.onnx").stat().st_size
