import json

import pytest

from scripts.diagnosis_eval import fit_prompt, parse_answer, render, summarize


def test_prompt_hides_labels_and_respects_budget():
    row = {"ts": 1, "service": "orders", "level": "ERROR", "message": "failed",
           "sim_fault": "secret", "target_service": "payments", "is_causal": True}
    line = render(row)
    assert set(json.loads(line)) == {"ts", "service", "level", "message"}
    assert "secret" not in line
    single, shown = fit_prompt([line], len, 1000)
    prompt, shown = fit_prompt([line] * 10, len, len(single))
    assert shown == 1
    assert len(prompt) <= len(single)


def test_empty_prompt_evidence_is_rejected():
    with pytest.raises(ValueError):
        fit_prompt(["x" * 1000], len, 500)


@pytest.mark.parametrize("text", ['{}', 'not json', '[]', '{"root_cause_service": ["orders"]}',
                                  '{"root_cause_service": "invented", "explanation": "x"}'])
def test_invalid_diagnoses_are_not_counted_as_valid(text):
    assert parse_answer(text) == {"valid": False, "service": "invalid"}


def test_summary_keeps_invalid_answers_in_denominator_and_excludes_cache_latency():
    rows = [{"arm": "raw", "correct": True, "valid": True, "tokens_in": 100,
             "tokens_out": 10, "shown_lines": 2, "candidate_lines": 10,
             "raw_lines": 10, "cached": False, "end_to_end_seconds": 4},
            {"arm": "raw", "correct": False, "valid": False, "tokens_in": 100,
             "tokens_out": 10, "shown_lines": 2, "candidate_lines": 10,
             "raw_lines": 10, "cached": True, "end_to_end_seconds": 0.01}]
    result = summarize(rows)["raw"]
    assert result["service_accuracy"] == 0.5
    assert result["valid_response_rate"] == 0.5
    assert result["mean_end_to_end_seconds"] == 4
    assert result["truncated_incidents"] == 2
