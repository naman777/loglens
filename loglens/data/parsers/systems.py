"""One small streaming parser per Loghub system, all yielding the common schema.

Schema: system, ts (epoch ms, UTC), host, component, level, message, session_id, label, line_no.
Parsers are generators over a raw directory so multi-GB files never sit in memory.
"""
from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path

Row = dict


def _ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _row(system, ts, host, component, level, message, session_id=None, label=None, line_no=0) -> Row:
    return {
        "system": system, "ts": ts, "host": host, "component": component, "level": level,
        "message": message, "session_id": session_id, "label": label, "line_no": line_no,
    }


def _lines(path: Path) -> Iterator[tuple[int, str]]:
    with open(path, "rb") as f:
        for i, raw in enumerate(f):
            try:
                yield i, raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                yield i, ""  # counted as dropped by the cleaner


# ---------------------------------------------------------------- BGL / Thunderbird
_BGL = re.compile(
    r"^(\S+) (\d+) (\S+) (\S+) (\d{4}-\d\d-\d\d-\d\d\.\d\d\.\d\d)\.(\d+) (\S+) (\S+) (\S+) (\S+) ?(.*)$"
)


def parse_bgl(raw: Path) -> Iterator[Row]:
    for i, line in _lines(raw / "BGL.log"):
        m = _BGL.match(line)
        if not m:
            yield _row("BGL", None, None, None, None, line, line_no=i)
            continue
        alert, _, _, node, t, us, _, _, comp, level, msg = m.groups()
        dt = datetime.strptime(t, "%Y-%m-%d-%H.%M.%S")
        yield _row("BGL", _ms(dt) + int(us[:6].ljust(6, "0")) // 1000, node, comp, level, msg,
                   None, 0 if alert == "-" else 1, i)


_TBIRD = re.compile(r"^(\S+) (\d+) (\S+) (\S+) (\w{3}) +(\d+) (\d\d:\d\d:\d\d) (\S+) (.*)$")


def parse_thunderbird(raw: Path) -> Iterator[Row]:
    for i, line in _lines(raw / "Thunderbird.log"):
        m = _TBIRD.match(line)
        if not m:
            yield _row("Thunderbird", None, None, None, None, line, line_no=i)
            continue
        alert, epoch, _, host, _mon, _day, _t, src, rest = m.groups()
        comp, _, msg = rest.partition(": ")
        if not msg:
            comp, msg = "", rest
        comp = re.sub(r"\[\d+\]$", "", comp)
        yield _row("Thunderbird", int(epoch) * 1000, host, comp, None, msg, None,
                   0 if alert == "-" else 1, i)


# ---------------------------------------------------------------- HDFS
_HDFS = re.compile(r"^(\d{6}) (\d{6}) (\d+) (\w+) ([^:]+): (.*)$")
_BLK = re.compile(r"blk_-?\d+")


def _hdfs_labels(raw: Path) -> dict[str, int]:
    p = raw / "preprocessed" / "anomaly_label.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {r["BlockId"]: int(r["Label"] == "Anomaly") for r in csv.DictReader(f)}


def parse_hdfs(raw: Path) -> Iterator[Row]:
    labels = _hdfs_labels(raw)
    for i, line in _lines(raw / "HDFS.log"):
        m = _HDFS.match(line)
        if not m:
            yield _row("HDFS", None, None, None, None, line, line_no=i)
            continue
        d, t, _pid, level, comp, msg = m.groups()
        dt = datetime.strptime(d + t, "%y%m%d%H%M%S")
        b = _BLK.search(msg)
        sid = b.group(0) if b else None
        yield _row("HDFS", _ms(dt), None, comp, level, msg, sid, labels.get(sid) if sid else None, i)


# ---------------------------------------------------------------- small systems
_APACHE = re.compile(r"^\[(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4})\] \[(\w+)\] (.*)$")


def parse_apache(raw: Path) -> Iterator[Row]:
    for i, line in _lines(raw / "Apache.log"):
        m = _APACHE.match(line)
        if not m:
            yield _row("Apache", None, None, None, None, line, line_no=i)
            continue
        dt = datetime.strptime(re.sub(r" +", " ", m.group(1)), "%a %b %d %H:%M:%S %Y")
        yield _row("Apache", _ms(dt), None, "httpd", m.group(2).upper(), m.group(3), line_no=i)


_ZK = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) - (\w+) +\[([^\]]*)\] - (.*)$")


def parse_zookeeper(raw: Path) -> Iterator[Row]:
    for i, line in _lines(raw / "Zookeeper.log"):
        m = _ZK.match(line)
        if not m:
            yield _row("Zookeeper", None, None, None, None, line, line_no=i)
            continue
        dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        comp = m.group(4).split(":", 1)[-1]
        yield _row("Zookeeper", _ms(dt) + int(m.group(2)), None, comp, m.group(3), m.group(5),
                   line_no=i)


_SYSLOG = re.compile(r"^(\w{3}) +(\d+) (\d\d:\d\d:\d\d) (\S+) ([^:]+?)(?:\[\d+\])?: (.*)$")


def _syslog(system: str, path: Path, year: int) -> Iterator[Row]:
    for i, line in _lines(path):
        m = _SYSLOG.match(line)
        if not m:
            yield _row(system, None, None, None, None, line, line_no=i)
            continue
        mon, day, t, host, comp, msg = m.groups()
        try:
            dt = datetime.strptime(f"{year} {mon} {day} {t}", "%Y %b %d %H:%M:%S")
        except ValueError:
            yield _row(system, None, host, comp, None, msg, line_no=i)
            continue
        yield _row(system, _ms(dt), host, comp, None, msg, line_no=i)


def parse_linux(raw: Path) -> Iterator[Row]:
    return _syslog("Linux", raw / "Linux.log", 2005)


def parse_ssh(raw: Path) -> Iterator[Row]:
    return _syslog("SSH", raw / "SSH.log", 2015)


_HPC = re.compile(r"^(\d+) (\S+) (\S+) (.+?) (\d{9,10}) (\d+) (.*)$")


def parse_hpc(raw: Path) -> Iterator[Row]:
    for i, line in _lines(raw / "HPC.log"):
        m = _HPC.match(line)
        if not m:
            yield _row("HPC", None, None, None, None, line, line_no=i)
            continue
        _id, a, b, state, epoch, _flag, msg = m.groups()
        host, comp = (a, b) if a.startswith("node") else (b, a)
        yield _row("HPC", int(epoch) * 1000, host, comp, None, f"{state} {msg}", line_no=i)


_OS = re.compile(r"^(\S+) (\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)\.(\d+) (\d+) (\w+) (\S+) (.*)$")
_OS_INST = re.compile(r"\[instance: ([0-9a-f-]{36})\]")


def parse_openstack(raw: Path) -> Iterator[Row]:
    n = 0
    for fname, label in (("openstack_normal1.log", 0), ("openstack_normal2.log", 0),
                         ("openstack_abnormal.log", 1)):
        p = raw / fname
        if not p.exists():
            continue
        for _, line in _lines(p):
            m = _OS.match(line)
            if not m:
                yield _row("OpenStack", None, None, None, None, line, line_no=n)
            else:
                _f, d, t, ms, _pid, level, comp, msg = m.groups()
                dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
                inst = _OS_INST.search(msg)
                yield _row("OpenStack", _ms(dt) + int(ms[:3].ljust(3, "0")), None, comp, level, msg,
                           inst.group(1) if inst else None, label, n)
            n += 1


_SPARK = re.compile(r"^(\d\d/\d\d/\d\d \d\d:\d\d:\d\d) (\w+) ([^:]+): (.*)$")


def parse_spark(raw: Path) -> Iterator[Row]:
    n = 0
    for app in sorted(p for p in raw.iterdir() if p.is_dir() and p.name.startswith("application_")):
        for f in sorted(app.glob("*.log")):
            for _, line in _lines(f):
                m = _SPARK.match(line)
                if m:
                    dt = datetime.strptime(m.group(1), "%y/%m/%d %H:%M:%S")
                    yield _row("Spark", _ms(dt), f.stem, m.group(3), m.group(2), m.group(4),
                               app.name, None, n)
                else:
                    yield _row("Spark", None, f.stem, None, None, line, app.name, None, n)
                n += 1


_HADOOP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) (\w+) \[([^\]]*)\] ([^:]+): (.*)$")


def _hadoop_labels(raw: Path) -> dict[str, int]:
    p = raw / "abnormal_label.txt"
    labels: dict[str, int] = {}
    if not p.exists():
        return labels
    cur = 0
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.endswith(":") and not s.startswith("+"):
            cur = 0 if s.lower().startswith("normal") else 1
        elif s.startswith("+ application_"):
            labels[s[2:].strip()] = cur
    return labels


def parse_hadoop(raw: Path) -> Iterator[Row]:
    labels = _hadoop_labels(raw)
    n = 0
    for app in sorted(p for p in raw.iterdir() if p.is_dir() and p.name.startswith("application_")):
        for f in sorted(app.glob("*.log")):
            for _, line in _lines(f):
                m = _HADOOP.match(line)
                lab = labels.get(app.name)
                if m:
                    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    yield _row("Hadoop", _ms(dt) + int(m.group(2)), f.stem, m.group(5), m.group(3),
                               m.group(6), app.name, lab, n)
                else:
                    yield _row("Hadoop", None, f.stem, None, None, line, app.name, lab, n)
                n += 1


def parse_mac(raw: Path) -> Iterator[Row]:
    return _syslog("Mac", raw / "Mac.log", 2017)


_HEALTH = re.compile(r"^(\d{8}-\d\d:\d\d:\d\d:\d+)\|([^|]*)\|([^|]*)\|(.*)$")


def parse_healthapp(raw: Path) -> Iterator[Row]:
    for i, line in _lines(raw / "HealthApp.log"):
        m = _HEALTH.match(line)
        if not m:
            yield _row("HealthApp", None, None, None, None, line, line_no=i)
            continue
        dt = datetime.strptime(m.group(1)[:17], "%Y%m%d-%H:%M:%S")
        yield _row("HealthApp", _ms(dt) + int(m.group(1)[18:]), m.group(3), m.group(2), None,
                   m.group(4), line_no=i)


_PROXIFIER = re.compile(r"^\[(\d\d\.\d\d \d\d:\d\d:\d\d)\] (\S+) - (.*)$")


def parse_proxifier(raw: Path) -> Iterator[Row]:
    for i, line in _lines(raw / "Proxifier.log"):
        m = _PROXIFIER.match(line)
        if not m:
            yield _row("Proxifier", None, None, None, None, line, line_no=i)
            continue
        dt = datetime.strptime("2015." + m.group(1), "%Y.%m.%d %H:%M:%S")
        yield _row("Proxifier", _ms(dt), None, m.group(2), None, m.group(3), line_no=i)


_ANDROID = re.compile(r"^(\d\d-\d\d \d\d:\d\d:\d\d)\.(\d+) +(\d+) +(\d+) ([VDIWEF]) ([^:]+): (.*)$")
_ALEVEL = {"V": "TRACE", "D": "DEBUG", "I": "INFO", "W": "WARN", "E": "ERROR", "F": "FATAL"}


def parse_android(raw: Path) -> Iterator[Row]:
    for i, line in _lines(raw / "Android.log"):
        m = _ANDROID.match(line)
        if not m:
            yield _row("Android", None, None, None, None, line, line_no=i)
            continue
        dt = datetime.strptime("2017-" + m.group(1), "%Y-%m-%d %H:%M:%S")
        yield _row("Android", _ms(dt) + int(m.group(2)[:3].ljust(3, "0")), m.group(3), m.group(6),
                   _ALEVEL[m.group(5)], m.group(7), line_no=i)


PARSERS: dict[str, Callable[[Path], Iterator[Row]]] = {
    "BGL": parse_bgl,
    "Thunderbird": parse_thunderbird,
    "HDFS": parse_hdfs,
    "Apache": parse_apache,
    "Zookeeper": parse_zookeeper,
    "Linux": parse_linux,
    "SSH": parse_ssh,
    "HPC": parse_hpc,
    "OpenStack": parse_openstack,
    "Spark": parse_spark,
    "Hadoop": parse_hadoop,
    "Mac": parse_mac,
    "HealthApp": parse_healthapp,
    "Proxifier": parse_proxifier,
    "Android": parse_android,
}
