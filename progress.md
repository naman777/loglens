# LogLens — Progress

Source of truth for what is done, in flight, and next. Plans: `LogLens — Build Plan.md` (why) and
`LogLens — Implementation Plan.md` (how).

Environment: Windows, Python 3.12, PyTorch 2.11 + CUDA, RTX 3050 6 GB (so pretraining is scaled down
vs. the plan's 4090/A100 assumption). No W&B key set → tracker falls back to local JSONL
(`results/runs/`), W&B used automatically when `wandb: true`.

## Status by phase

| Phase | Status |
| --- | --- |
| 0 Setup | in progress |
| 1 Data pipeline | not started |
| 2 Fault-injection lab | not started |
| 3 Tokenizer | not started |
| 4 Line encoder pretraining | not started |
| 5 Window model + heads | not started |
| 6 Baselines + eval | not started |
| 7 CPU serving | not started |
| 8 Integration + release | not started |

## Log

- 2026-09-30: repo initialised, GitHub repo `naman777/loglens` created, skeleton + config/seed/tracker.

## Deviations / honest notes

(none yet)
