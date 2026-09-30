import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import make_release


def test_custom_release_destination_and_checksums(monkeypatch, tmp_path):
    root = tmp_path / "source"
    (root / "docs").mkdir(parents=True)
    (root / "docs/model_card.md").write_text("Card")
    (root / "model.bin").write_bytes(b"model")
    monkeypatch.setattr(make_release, "ROOT", root)
    monkeypatch.setattr(make_release, "OUT", tmp_path / "must-not-use")
    monkeypatch.setattr(make_release, "FILES", {"model.bin": "onnx/model.bin"})
    monkeypatch.setattr(make_release.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="commit"))
    out = tmp_path / "chosen"
    make_release.main(out)
    assert not (tmp_path / "must-not-use").exists()
    assert (out / "onnx/model.bin").read_bytes() == b"model"
    manifest = json.loads((out / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        data = (out / name).read_bytes()
        assert entry == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    with pytest.raises(FileExistsError):
        make_release.main(out)


def test_missing_release_artifacts_do_not_create_destination(monkeypatch, tmp_path):
    monkeypatch.setattr(make_release, "ROOT", tmp_path)
    out = tmp_path / "release"
    with pytest.raises(SystemExit, match="missing/empty"):
        make_release.main(out)
    assert not out.exists()
