# LogLens — Progress

Source of truth for what is done, in flight, and next. Plans: `docs/plans/build_plan.md` (why) and
`docs/plans/implementation_plan.md` (how). Repo: https://github.com/naman777/loglens

Environment: Windows 11, Python 3.12, PyTorch 2.11 + CUDA, RTX 3050 6 GB laptop GPU, 16 GB RAM
(~4 GB free during big builds), no Docker, no W&B key (tracker falls back to local JSONL in
`results/runs/`; W&B is used automatically with `wandb: true`).

## Status by phase

| Phase | Status |
| --- | --- |
| 0 Setup | done |
| 1 Data pipeline | done (16 systems; Thunderbird/Spark partial, see deviations) |
| 2 Fault-injection lab | done via **simulator** (180 incidents, 10 fault types); Docker version written, untested |
| 3 Tokenizer | done; masking gate/2x-token gate **not met** (see below) |
| 4 Line encoder pretraining | re-running after masking fix (`configs/pretrain.yaml`, ~25k steps) |
| 5 Window model + heads | code done + smoke-tested; real training waits for the encoder |
| 6 Baselines + eval | harness, Drain3+DeepLog, IsolationForest, heuristics, LLM harness written; runs pending |
| 7 CPU serving | export (parity+int8), cache, runtime, API, CLI, Dockerfile written; benchmark pending |
| 8 Integration + release | agent tool written; model card / README table / blog pending |

## Log

- Repo initialised, GitHub repo `naman777/loglens`, skeleton, config/seed/tracker, CI (ruff + pytest).
- Loghub downloaded from Zenodo (record 8196385) with md5 verification; parsers for 15 systems +
  synthetic lab; time-split Parquet builder with session-atomic splits; manifest hash.
- Masking rules (79 unit tests), BPE (8k/16k) trainer, tokenizer comparison report.
- Simulated lab: 10 fault types, 10 campaigns x 18 incidents = 180 incidents, rule-labelled causal
  lines (F1 0.94 vs simulator ground truth).
- LineEncoder (33.6M params @ 16k vocab), MLM loop with token-budget batching, embed/pretok tools.
- Window model (4 layers, 256-d) + anomaly / suspicion / template heads; unsupervised and supervised
  training; metrics; classic baselines; ONNX export with FP32-vs-ONNX parity < 1e-3 and dynamic int8.
- Found and fixed a masking flaw: 65,536 distinct `generating core.N` BGL lines (36% of BGL lines)
  were not being collapsed. Numeric suffixes in identifiers (`core.N`, `step_N`, `bglio12`) are now
  masked; data, tokenizer and pretraining re-run from scratch.

## Deviations / honest notes

- **No Docker on this machine.** The fault-injection lab is a deterministic Python simulator
  (`lab/sim/engine.py`) reproducing the 5-service app's log behaviour, with the same fault taxonomy
  and label schema. Compose files, service code, fault injector and Locust file exist under `lab/`
  but were **not run**. All RCA numbers are on *synthetic* incidents; every table says so.
- **Hand check replaced by ground-truth agreement**: the causal-line rule labeller is validated against
  the simulator's own per-line ground truth (F1 0.94) instead of manual review.
- **Thunderbird**: only the first 6M lines (of ~211M) streamed from the archive; the archive checksum
  cannot be verified for a truncated stream. It is the unseen system (never used for tokenizer,
  encoder or window pretraining). Its anomaly windows are 20 lines (256-line windows are 96% positive
  in val/test because alerts are ~5-8% of lines).
- **Spark** capped at 6M lines (of ~33M) for RAM; **HDFS** uses `HDFS_v1` labels per block.
- **Distinct masked lines are tiny**: ~135k distinct masked lines in total (BGL 70k pre-fix). So the
  "0.5-1B token" pretraining budget of the plan is not meaningful; pretraining samples *distinct*
  lines (per-system share capped) and runs a fixed number of steps.
- **GPU**: RTX 3050 6 GB. Pretraining batches are token-budgeted (16k tokens) to stay out of shared
  memory.
- **Tokenizer gate failed**: ratio of raw-GPT-2 tokens to LogLens content tokens is 1.45x on average and
  1.26x on the unseen system (gate was >= 2x). Masking is a wash on short lines and helps on
  ID-heavy ones (HDFS 2.3x, OpenStack 3.9x). Reported as-is in `docs/tokenizer_report.md`.
  16k vocab chosen by the plan's rule (8k was not within 5% of 16k).
- BGL/Thunderbird windows: BGL 256/128 sliding, HDFS per-block sessions, Thunderbird 20/20.
- Frontier-LLM API baseline: **cannot be run here** (no API key in this environment); harness and
  caching are implemented; the table will show "not run" rather than a fabricated number.
- "Vocabulary is masked-template-based" (top-2000 masked lines) rather than Drain3 clusters for the
  unsupervised template targets; Drain3 is used for the baselines and the data report.

## Next

1. Finish encoder pretraining -> embeddings -> embedding sanity report.
2. Window model: unsupervised, supervised, ablations (gap embedding, random-init encoder).
3. `make eval` benchmark table; small open LLM baseline on the lab test incidents.
4. ONNX export, int8 accuracy, CPU throughput benchmark, model card, README table.
