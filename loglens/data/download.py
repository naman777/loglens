"""Fetch Loghub archives from Zenodo, verify checksums, extract to data/raw/<system>/."""
from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "8196385"  # Loghub (full datasets)
API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# system -> archive key on Zenodo
ARCHIVES = {
    "Thunderbird": "Thunderbird.tar.gz",
    "HDFS": "HDFS_v1.zip",
    "BGL": "BGL.zip",
    "OpenStack": "OpenStack.tar.gz",
    "Hadoop": "Hadoop.zip",
    "Spark": "Spark.tar.gz",
    "Zookeeper": "Zookeeper.tar.gz",
    "Linux": "Linux.tar.gz",
    "Apache": "Apache.tar.gz",
    "HPC": "HPC.zip",
    "SSH": "SSH.tar.gz",
    "Mac": "Mac.tar.gz",
    "HealthApp": "HealthApp.tar.gz",
    "Proxifier": "Proxifier.tar.gz",
    "Android": "Android_v1.zip",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_files() -> dict[str, dict]:
    with urllib.request.urlopen(API, timeout=60) as r:
        rec = json.load(r)
    return {f["key"]: f for f in rec["files"]}


def fetch(system: str, raw_dir: str | Path = "data/raw", files: dict | None = None) -> Path:
    raw = Path(raw_dir)
    out = raw / system
    if out.exists() and any(out.iterdir()):
        return out
    files = files or record_files()
    meta = files[ARCHIVES[system]]
    arc = raw / "_archives" / meta["key"]
    arc.parent.mkdir(parents=True, exist_ok=True)
    if not arc.exists():
        url = meta["links"]["self"]
        print(f"downloading {system}: {meta['key']} ({meta['size'] / 1e6:.0f} MB)")
        urllib.request.urlretrieve(url, arc)
    expected = meta["checksum"].split(":", 1)[-1]
    if _md5(arc) != expected:
        arc.unlink()
        raise RuntimeError(f"checksum mismatch for {arc.name}")
    out.mkdir(parents=True, exist_ok=True)
    if arc.suffix == ".zip":
        with zipfile.ZipFile(arc) as z:
            z.extractall(out)
    else:
        with tarfile.open(arc) as t:
            t.extractall(out)
    return out


def main(systems: list[str] | None = None) -> None:
    files = record_files()
    for s in systems or [k for k in ARCHIVES if k != "Thunderbird"]:
        print(s, "->", fetch(s, files=files))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or None)
