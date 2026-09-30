"""Stream the first N lines of a huge Loghub tar.gz (e.g. Thunderbird) without downloading it all.

The archive checksum cannot be verified for a truncated stream; the line count is recorded instead.
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

from loglens.data.download import ARCHIVES, record_files


def fetch_partial(system: str, max_lines: int, raw_dir: str | Path = "data/raw") -> Path:
    out = Path(raw_dir) / system / f"{system}.log"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = record_files()[ARCHIVES[system]]
    n = 0
    with urllib.request.urlopen(meta["links"]["self"], timeout=120) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                with open(out, "wb") as w:
                    for line in f:
                        w.write(line)
                        n += 1
                        if n >= max_lines:
                            break
                break
    print(f"{system}: kept {n} lines -> {out}")
    return out


if __name__ == "__main__":
    fetch_partial(sys.argv[1], int(sys.argv[2]))
