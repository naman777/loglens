import pytest

from loglens.config import load_config
from loglens.train.dummy import DummyConfig


def test_config_defaults_and_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("steps: 5\n")
    assert load_config(DummyConfig, p).steps == 5
    p.write_text("bogus: 1\n")
    with pytest.raises(ValueError):
        load_config(DummyConfig, p)


def test_dummy_run_learns(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from loglens.train.dummy import run

    monkeypatch.chdir(tmp_path)
    losses = run(DummyConfig(steps=100))
    assert losses[-1] < losses[0] * 0.7
