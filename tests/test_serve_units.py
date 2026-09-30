import numpy as np

from loglens.serve.cache import EmbeddingCache
from loglens.serve.runtime import parse_line


def test_cache_lru_and_hit_rate():
    c = EmbeddingCache(capacity=2)
    a, b, d = (np.ones(4, np.float32) * i for i in range(3))
    assert c.get(1) is None
    c.put(1, a)
    c.put(2, b)
    assert c.get(1) is not None  # 1 becomes most recent
    c.put(3, d)  # evicts 2
    assert c.get(2) is None and c.get(3) is not None and len(c) == 2
    assert c.hits == 2 and c.misses == 2 and c.hit_rate == 0.5


def test_parse_json_and_plain_lines():
    j = parse_line('{"ts": 1772323200379, "service": "orders", "level": "WARN", "message": "slow"}')
    assert (j.ts, j.service, j.level, j.message) == (1772323200379, "orders", "WARN", "slow")
    p = parse_line("2026-03-01 10:00:01,250 ERROR [payments] bank timeout after 5000ms")
    assert p.level == "ERROR" and p.service == "payments" and p.ts is not None
    q = parse_line("Jun  9 06:06:20 combo sshd[1234]: Failed password")
    assert q.service == "sshd" and q.level == "UNK"
    assert parse_line("plain text with no structure").service == "other"


def test_line_formats_match_training_parsers(tmp_path):
    from loglens.data.parsers.systems import parse_bgl, parse_hdfs
    from loglens.serve.formats import LINE_FORMATS

    bgl = ("- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 "
           "RAS KERNEL INFO instruction cache parity error corrected")
    (tmp_path / "BGL.log").write_text(bgl + "\n")
    (ref,) = list(parse_bgl(tmp_path))
    got = LINE_FORMATS["bgl"](bgl)
    assert (got.message, got.level, got.service) == (ref["message"], ref["level"], "BGL")
    hdfs = "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 src: /1.2.3.4:5"
    (tmp_path / "HDFS.log").write_text(hdfs + "\n")
    (ref,) = list(parse_hdfs(tmp_path))
    got = LINE_FORMATS["hdfs"](hdfs)
    assert (got.message, got.level, got.service) == (ref["message"], ref["level"], "HDFS")
