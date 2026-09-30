# Paired LLM diagnosis evaluation

Run: 2026-09-30; Qwen/Qwen2.5-1.5B-Instruct, local GPU, greedy generation, max 128 output tokens.
Full evidence: `results/diagnosis/20260930T101443431824Z/` (metadata, 72 answers and summary).

All **36 held-out synthetic lab incidents** from campaigns c08/c09 were evaluated in both arms.
The prompt asks for a root-cause service plus explanation. Only timestamp, service, level and
message enter the prompt; simulator fault annotations and target labels are removed. Invalid JSON
or schema violations count as incorrect. Arm order is shuffled per incident with seed 1337.

| Metric | Chronological raw prefix | LogLens top 15 |
| --- | --- | --- |
| Correct root-cause service | 6/36 (16.7%) | 31/36 (86.1%) |
| Valid structured answers | 33/36 (91.7%) | 35/36 (97.2%) |
| Mean actual input tokens | 1,503.6 | 919.7 |
| Mean output tokens | 38.8 | 39.9 |
| Mean end-to-end seconds | 4.70 | 4.75 |
| Incidents truncated after candidate selection | 36/36 | 0/36 |
| Mean fraction of original lines shown | 1.58% | 0.88% |

LogLens selection improved service accuracy by 69.4 percentage points in this setting and used
about 38.8% fewer input tokens. **No latency improvement was established.**

## Interpretation and limits

- This is actual LLM answer scoring, beyond the earlier causal-line/token-count proxy. It is still
  a single-call diagnosis experiment, not an autonomous agent using tools across multiple steps.
- Both arms use the same **1,500 text-token budget**; chat-template overhead is additional and is
  included in the reported provider input-token counts. The raw-prefix arm sees only the earliest
  1.58% of lines on average. This deliberately tests selection under a tight budget; it does not
  compare against full-context, chunked, retrieval-based or larger-model diagnosis.
- Ground truth is the simulator's target service. LogLens was trained on other campaigns from the
  same generator. These results do not prove production root-cause accuracy or generalization.
- One local small model, one prompt, one decoding setup and 36 cases: no broad LLM superiority claim.
- Timing excludes model loading, includes selection/prompt preparation and generation, and reuses
  the LogLens runtime cache across incidents. All 72 calls were uncached. Laptop timings varied
  substantially late in the run; paired random ordering helps but is not a controlled latency study.
- The GPU environment used the existing system Python and emitted the known NumPy/SciPy warning.
  Clean-environment code tests pass separately; the new CPU `.venv` was not used for GPU generation.

## Reproduce

With matching ONNX/tokenizer artifacts, synthetic campaign logs, held-out incident Parquet, and
the `llm` extra installed:

```bash
python scripts/diagnosis_eval.py --max-prompt-tokens 1500 --max-new-tokens 128
```

Set `HF_HUB_OFFLINE=1` when reusing already downloaded Hugging Face weights. Each run creates a new
output directory. A `--limit` run is a seeded pilot subset and must not be reported as all 36 cases.
API-provider token budgeting is approximate; inspect exact returned token usage before comparing
costs. The harness does not assign current API prices or claim a frontier-model result.
