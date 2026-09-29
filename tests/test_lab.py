import json

from lab.run_campaign import run_one
from lab.sim.engine import FAULT_TYPES
from loglens.data.lab import label_campaign, load_campaign


def test_campaign_is_deterministic(tmp_path):
    a = run_one(7, "a", 3, tmp_path, 0)
    b = run_one(7, "b", 3, tmp_path, 0)
    assert a == b
    assert (tmp_path / "a" / "logs.jsonl").read_text() == (tmp_path / "b" / "logs.jsonl").read_text()


def test_every_fault_type_has_causal_lines_and_rule_agrees(tmp_path):
    run_one(11, "c", len(FAULT_TYPES) * 2, tmp_path, 0)
    lines, incs = load_campaign(tmp_path / "c")
    df, incs, agree = label_campaign(lines, incs)
    assert {i["fault_type"] for i in incs} == set(FAULT_TYPES)
    by = {}
    for i, a in zip(incs, agree, strict=True):
        by.setdefault(i["fault_type"], 0)
        by[i["fault_type"]] += a["tp"]
    assert all(v > 0 for v in by.values()), by
    tp = sum(a["tp"] for a in agree)
    fp = sum(a["fp"] for a in agree)
    fn = sum(a["fn"] for a in agree)
    assert 2 * tp / (2 * tp + fp + fn) > 0.85
    # causal lines only come from each incident's target service
    for inc in incs:
        sub = df.filter((df["incident_id"] == inc["incident_id"]) & df["is_causal"])
        assert set(sub["service"]) <= {inc["target_service"]}
    json.dumps(incs)
