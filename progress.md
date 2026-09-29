# LogLens — Progress

Source of truth for what is done, in flight, and next. Plans: `LogLens — Build Plan.md` (why) and
`LogLens — Implementation Plan.md` (how).

Environment: Windows, Python 3.12, PyTorch 2.11 + CUDA, RTX 3050 6 GB (so pretraining is scaled down
vs. the plan's 4090/A100 assumption). No W&B key set → tracker falls back to local JSONL
(`results/runs/`), W&B used automatically when `wandb: true`.

## Status by phase

| Phase | Status |
| --- | --- |
| 0 Setup | done (W&B not configured; local JSONL tracker fallback) |
| 1 Data pipeline | in progress: downloads, parsers, builder done; full build running |
| 2 Fault-injection lab | done via simulator (180 incidents, 10 fault types); Docker lab files pending |
| 3 Tokenizer | masking done + tested; BPE trainer/eval written, awaiting masked data |
| 4 Line encoder pretraining | model (29.5M params) + MLM loop written, awaiting tokenizer/data |
| 5 Window model + heads | not started |
| 6 Baselines + eval | not started |
| 7 CPU serving | not started |
| 8 Integration + release | not started |

## Log

- 2026-09-30: repo initialised, GitHub repo `naman777/loglens` created, skeleton + config/seed/tracker.

- 2026-09-30: Loghub downloaded (Zenodo 8196385), parsers for 11 systems, time-split Parquet builder.
- 2026-09-30: Masking rules (71 tests), BPE trainer, tokenizer evaluation script.
- 2026-09-30: Simulated lab: 10 fault types, 10 campaigns x 18 incidents = 180, causal-line rule labeller
  (F1 0.94 vs simulator ground truth).
- 2026-09-30: LineEncoder 29.5M params, MLM pretraining loop, metrics module.

## Deviations / honest notes

- **No Docker on this machine.** The fault-injection lab is a deterministic Python simulator
  (`lab/sim/engine.py`) reproducing the 5-service app's log behaviour, with the same fault taxonomy and
  label schema. Docker Compose files are provided for a real deployment but were not run here.
  Results on lab data are therefore on *synthetic* incidents -- state this everywhere it is quoted.
- **Hand-check replaced by ground-truth agreement**: the rule labeller is validated against the
  simulator's own per-line ground truth instead of manual checking (no human labeller available).
- **Thunderbird**: only the first 6M lines (of ~211M) streamed from the archive; archive checksum not
  verifiable for a truncated stream.
- **Spark** capped at 6M lines (of ~33M) for RAM (16 GB laptop, ~4 GB free during builds).
- **GPU**: RTX 3050 6 GB, so pretraining budget is far below the plan's 0.5-1B tokens on a 4090/A100.
