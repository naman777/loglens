from pathlib import Path

import polars as pl

from loglens.data.build import DataConfig, assign_splits, check_no_overlap
from loglens.data.parsers.systems import parse_bgl, parse_hdfs, parse_zookeeper


def _write(tmp: Path, name: str, text: str) -> Path:
    (tmp / name).write_text(text)
    return tmp


def test_bgl_parser_labels_and_time(tmp_path):
    _write(tmp_path, "BGL.log",
           "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 "
           "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected\n"
           "KERNDTLB 1117838571 2005.06.03 R02 2005-06-03-15.42.51.000000 R02 RAS KERNEL FATAL boom\n")
    a, b = list(parse_bgl(tmp_path))
    assert (a["label"], b["label"]) == (0, 1)
    assert a["level"] == "INFO" and a["component"] == "KERNEL"
    assert a["message"] == "instruction cache parity error corrected"
    assert a["ts"] == 1117813370363  # 2005-06-03 15:42:50.363 UTC


def test_hdfs_session_and_label(tmp_path):
    _write(tmp_path, "HDFS.log",
           "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 src: /1.2.3.4:5\n")
    (tmp_path / "preprocessed").mkdir()
    (tmp_path / "preprocessed" / "anomaly_label.csv").write_text("BlockId,Label\nblk_-16,Anomaly\n")
    (r,) = list(parse_hdfs(tmp_path))
    assert r["session_id"] == "blk_-16" and r["label"] == 1 and r["level"] == "INFO"


def test_zookeeper_parser(tmp_path):
    _write(tmp_path, "Zookeeper.log",
           "2015-07-29 17:41:41,536 - INFO  [main:QuorumPeerConfig@101] - Reading configuration\n")
    (r,) = list(parse_zookeeper(tmp_path))
    assert r["component"] == "QuorumPeerConfig@101" and r["message"] == "Reading configuration"


def test_split_is_time_ordered_and_session_atomic():
    n = 100
    df = pl.DataFrame({
        "ts": list(range(n)),
        "session_id": [f"s{i // 10}" for i in range(n)],  # sessions of 10 consecutive lines
    })
    out = assign_splits(df, DataConfig())
    per_session = out.group_by("session_id").agg(pl.col("split").n_unique())
    assert per_session["split"].max() == 1
    mx = out.filter(pl.col("split") == "train")["key_ts"].max()
    mn = out.filter(pl.col("split") == "test")["key_ts"].min()
    assert mx < mn
    check_no_overlap({"systems": {"x": {"splits": {
        "train": {"key_ts_max": mx}, "test": {"key_ts_min": mn}}}}})
