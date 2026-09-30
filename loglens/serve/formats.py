"""Per-format header stripping for the runtime, mirroring the training-time parsers.

The model was trained on the *message part* of each line with the system name as service token
(``<SVC:BGL>``). For those formats pass ``line_format='bgl'`` (etc.) so the runtime strips the same
header and uses the same service token; ``generic`` keeps the whole line and guesses the service.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from loglens.data.parsers import systems as S


@dataclass
class Parsed:
    ts: int | None
    level: str
    service: str
    message: str


def _bgl(line: str) -> Parsed | None:
    m = S._BGL.match(line)
    if not m:
        return None
    return Parsed(None, m.group(10), "BGL", m.group(11))


def _hdfs(line: str) -> Parsed | None:
    m = S._HDFS.match(line)
    return Parsed(None, m.group(4), "HDFS", m.group(6)) if m else None


def _thunderbird(line: str) -> Parsed | None:
    m = S._TBIRD.match(line)
    if not m:
        return None
    comp, _, msg = m.group(9).partition(": ")
    return Parsed(None, "UNK", "Thunderbird", msg if msg else m.group(9))


def _zookeeper(line: str) -> Parsed | None:
    m = S._ZK.match(line)
    return Parsed(None, m.group(3), "Zookeeper", m.group(5)) if m else None


def _apache(line: str) -> Parsed | None:
    m = S._APACHE.match(line)
    return Parsed(None, m.group(2).upper(), "Apache", m.group(3)) if m else None


def _syslog(name: str) -> Callable[[str], Parsed | None]:
    def f(line: str) -> Parsed | None:
        m = S._SYSLOG.match(line)
        return Parsed(None, "UNK", name, m.group(6)) if m else None
    return f


def _openstack(line: str) -> Parsed | None:
    m = S._OS.match(line)
    return Parsed(None, m.group(6), "OpenStack", m.group(8)) if m else None


def _android(line: str) -> Parsed | None:
    m = S._ANDROID.match(line)
    return Parsed(None, S._ALEVEL[m.group(5)], "Android", m.group(7)) if m else None


def _healthapp(line: str) -> Parsed | None:
    m = S._HEALTH.match(line)
    return Parsed(None, "UNK", "HealthApp", m.group(4)) if m else None


def _proxifier(line: str) -> Parsed | None:
    m = S._PROXIFIER.match(line)
    return Parsed(None, "UNK", "Proxifier", m.group(3)) if m else None


LINE_FORMATS: dict[str, Callable[[str], Parsed | None]] = {
    "bgl": _bgl, "hdfs": _hdfs, "thunderbird": _thunderbird, "zookeeper": _zookeeper,
    "apache": _apache, "linux": _syslog("Linux"), "ssh": _syslog("SSH"), "mac": _syslog("Mac"),
    "openstack": _openstack, "android": _android, "healthapp": _healthapp, "proxifier": _proxifier,
}
