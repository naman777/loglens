# LogLens — Progress

Source of truth for what is done, in flight, and next. Plans: `docs/plans/build_plan.md` (why) and
`docs/plans/implementation_plan.md` (how). Repo: https://github.com/naman777/loglens

Environment: Windows 11, Python 3.12, PyTorch 2.11 + CUDA, RTX 3050 6 GB laptop GPU, 16 GB RAM
(~4 GB free during big builds), 16 logical cores, **no Docker**, no W&B key (tracker falls back to
local JSONL in `results/runs/`; W&B is used automatically with `wandb: true`).

## Status by phase

| Phase | Status | Exit gate |
| --- | --- | --- |
| 0 Setup | done | met (CI = ruff + pytest; W&B not configured, local tracker instead) |
| 1 Data pipeline | done: 15 Loghub systems + lab in Parquet, time splits, manifest, `docs/data_report.md` | met (no train/test key-time overlap); Thunderbird/Spark are capped subsets |
| 2 Fault-injection lab | done via **simulator**: 180 incidents, 10 fault types, causal lines labelled | count met; Docker version written but never run; "hand-check" replaced by agreement with simulator ground truth (F1 0.94) |
| 3 Tokenizer | done, 16k vocab | **not met**: 1.5x fewer tokens than GPT-2 (1.3x on unseen), gate was 2x; <2% of lines >128 tokens: met |
| 4 Line encoder | done (33.6M params, MLM val loss 0.83) | loss plateaued: met; NN template agreement 68% (gate 90%): **not met**; linear probe beats TF-IDF: **not met** |
| 5 Window model | done | beats Drain3+DeepLog on BGL F1 (0.954 vs 0.335, supervised vs unsupervised): met; lab RCA Recall@5 >= 0.6: met (0.94) |
| 6 Baselines + eval | done: `make eval` -> `results/benchmark.md` | filled except frontier-LLM row (no API key). Baselines: Drain3+DeepLog, Drain3+IsolationForest, Drain3+LogReg (supervised), heuristics, Qwen2.5-1.5B |
| 7 CPU serving | done | int8 within 0.5 F1 of FP32: met; 20k lines/s on 4 cores: **missed** (17.7k; 24k on 8 cores); `docker run` untested |
| 8 Integration + release | drafts done: agent tool + agent-side eval, demo script, README, model card, blog draft, release script | HF upload, video, Show HN, cold-email items **not done** (need the user's accounts); stranger-repro test not done |

## What is in the repo

Everything in the Implementation Plan's layout, plus `scripts/` (ablations, LLM eval, serving bench,
agent eval, error analysis, demo, consistency check, release packaging) and `docs/` reports.
Tests: 125 (masking, parsers, lab lifecycle/services, diagnosis evaluation, encoder, metrics, ONNX parity, runtime end-to-end).

## Key results (see README / results/*.md for tables)

- Lab RCA (36 synthetic test incidents): Recall@5 0.94, Recall@1 0.61, MRR 0.76.
- BGL F1 0.954, HDFS F1 0.965 (saturated: LogReg 0.969), Thunderbird zero-shot fails; 1% labels -> F1 0.985.
- Agent-side: ~140x fewer tokens (2,342 lines -> 15 lines), causal line in top 15 for 94%.
- int8 model 37.5 MB, <= 0.5 F1 drop; 17.7k lines/s pinned to 4 cores.
- Ablations: no measurable gain from encoder pretraining or the time-gap embedding in-distribution.

## Log

- Repo initialised, GitHub repo `naman777/loglens`, skeleton, config/seed/tracker, CI.
- Loghub from Zenodo (md5-verified), parsers, time-split builder; masking (80+ tests); BPE; simulated lab.
- Line encoder + MLM loop; window model + heads; supervised/unsupervised training; baselines; eval harness.
- Found & fixed a masking flaw (65,536 distinct `core.N` BGL lines); re-ran data/tokenizer/pretraining.
- SimCSE encoder tried: worse embeddings; kept plain MLM encoder.
- ONNX export (parity ~2e-6), dynamic int8, cache, pipelined runtime (parse+mask in workers), API/CLI.
- Found train/serve skew on unknown formats; added per-format header stripping + consistency report.

## Deviations / honest notes

- **No Docker on this machine.** The lab is a deterministic Python simulator reproducing the 5-service
  app's log behaviour with the same fault taxonomy and label schema. Compose files, service code,
  fault injector and Locust file exist under `lab/` but were **not run**. All RCA numbers are on
  *synthetic* incidents and the model was trained on the same generator.
- **Hand check replaced by ground-truth agreement**: causal-line rules validated against the
  simulator's per-line ground truth (F1 0.94).
- **Thunderbird**: only the first 6M lines (of ~211M) streamed; archive checksum cannot be verified for a
  truncated stream. It is the unseen system (never used for tokenizer/encoder/window pretraining). Its
  anomaly windows are 20 lines (256-line windows are 96% positive in val/test).
- **Spark** capped at 6M lines (of ~33M) for RAM; HDFS uses `HDFS_v1` block labels; OpenStack/Hadoop
  labels are file/app level and not used for supervised eval.
- **Distinct masked lines are tiny** (~59k train lines across 14 systems), so the plan's 0.5-1B-token
  pretraining is not meaningful; ran 12k steps on distinct lines (per-system share capped at 30%).
- **Tokenizer sample cap** was relaxed to 50% per system (30% would discard 80% of BGL's distinct lines).
- **GPU nondeterminism**: DeepLog baseline numbers move by a few points between runs (cuDNN LSTM), other
  methods are seeded.
- **Thresholds** are tuned on the validation split only; per-system (lab 0.84, BGL 0.89, HDFS 0.42 on
  sigmoid scale). The runtime default (0.85) is a compromise, not tuned for arbitrary systems.
- **Frontier-LLM baseline not run** (no API key); the Qwen2.5-1.5B row uses a 1,500-token prompt and
  sees ~2.5% of an incident. No cost table is claimed.
- Template targets for the unsupervised objective are top-2000 *masked lines*, not Drain3 clusters
  (Drain3 is used for baselines/reports).
- W&B not used (no account); local JSONL tracker records config, git hash, manifest hash per run.
- Not done: Hugging Face upload, 2-minute video, LinkedIn/HN posts, cold emails, the "friend follows the
  quickstart" test, `v1.0.0` tag (this is v0.1). `scripts/make_release.py` prepares the HF folder.

## Next (if continuing)

1. Verify the full training quickstart from a fresh clone and publish downloadable weights after review.
   Clean local installation of the dev/serve/export/baselines extras and saved-model smoke checks pass.
2. Run and validate the real Docker lab on a Docker-enabled host or CI. Six faults now have real
   dependency interactions and an evidence-producing runner, but Docker is unavailable locally.
   See `docs/real_lab.md` and `docs/next_steps.md` for the remaining validation work.
3. Evaluate few-label adaptation on another unseen system and actual LLM-agent diagnoses with/without
   LogLens. Existing agent-side token counts and service-vote metrics are proxies, not agent answers.
4. Improve unfamiliar-format parsing and threshold stability; then optimize the parse/mask hot path.

## Completed follow-up experiments (latest commit c260274)

- Duration masking ablation is complete: removing buckets reduced BGL F1 0.939 -> 0.872 and
  HDFS 0.995 -> 0.950 with a random-init encoder. Keep duration buckets.
- Template-contrastive encoder v3 is complete: nearest-neighbour agreement 72.3% (90% gate still
  missed); linear probe still loses to TF-IDF. Thunderbird PR-AUC rose to 0.700 and lab Recall@5
  to 1.000, but BGL F1 fell to 0.473 despite PR-AUC 0.962. Single seed; not adopted as main model.
- See `docs/error_analysis.md` and `docs/embedding_report_v3_template.md` for interpretation.

## Reproducibility follow-up (2026-09-30)

- Applied the previously blocked dependency, CI, quickstart, smoke-check, and release-packaging changes.
- Existing environment: all 100 tests and Ruff pass; saved FP32/int8 inference and cache reuse pass.
- Built `release/loglens-v0.1` locally and independently verified all 15 SHA-256 manifest entries.
  Packaging rejects missing artifacts and refuses to overwrite an existing release directory.
- Saved the real-lab fixes and experiment prerequisites in `docs/next_steps.md`. Docker remains
  unavailable; no real-container results or new accuracy measurements were produced. Nothing published.
- Fresh isolated `.venv`: installed `.[dev,serve,export,baselines]`; `pip check` reports no broken
  requirements, Ruff passes, all 100 tests pass, and both saved FP32/int8 smoke checks pass.
  NumPy/SciPy compatibility warning is absent there; existing ONNX export warnings remain.
  This validates a clean local install with existing artifacts, not a fresh-clone training run.

## Real-lab and diagnosis follow-up (2026-09-30)

- Implemented Postgres persistence, Redis task enqueue/consumption, payment connection holds and
  release, worker retry without premature acknowledgement, and gateway error propagation.
- Replaced unverified Docker fault helpers with six reversible contexts. Network latency, packet
  loss, CPU hog and memory leak remain simulator-only, explicitly unsupported by the real runner.
- Added health checks, 16 MiB tmpfs spool, resource limits, and a normal/fault/recovery runner that
  stores evidence separately under `lab/real-out/`. Invalid/unrecovered incidents are not accepted.
- Added Docker CI coverage with evidence upload (configured but not executed here).
- Local validation: all 125 tests pass, Ruff passes, `pip check` passes. Docker preflight fails
  because the executable is absent. No real-container results or manually reviewed labels claimed.
- Ran paired Qwen2.5-1.5B single-call diagnosis on all 36 held-out synthetic incidents: raw prefix
  6/36 correct services (16.7%), LogLens selection 31/36 (86.1%). Both use a 1,500 text-token budget
  plus chat formatting. Mean input tokens 1,503.6 vs 919.7; mean end-to-end time 4.70 vs 4.75 seconds.
  No latency improvement established. Full answers and limitations: `docs/diagnosis_eval.md`.
