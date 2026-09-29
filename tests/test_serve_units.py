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
