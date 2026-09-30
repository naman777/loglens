# LogLens

A small log-intelligence model, trained from scratch, that flags incidents and ranks the log lines
that explain them — on a laptop CPU, with zero LLM API calls.

> **Status: v0.1 (research prototype).** Everything below was produced by the scripts in this repo on
> one Windows laptop (RTX 3050 6 GB, no Docker). Read [Limitations](#limitations-read-this) before
> quoting any number — the root-cause results are on **synthetic** incidents, and zero-shot transfer
> to an unseen log system **did not work**.

## Headline results

| What | Result |
| --- | --- |
| Root-cause ranking, 36 synthetic lab incidents (10 fault types) | **Recall@5 0.94**, Recall@1 0.61, MRR 0.76 (Drain3+DeepLog surprise: 0.81 / 0.53 / 0.65; random: 0.17) |
| What an on-call agent has to read | **~140x fewer tokens**: 2,342 lines (40k GPT-2 tokens) -> top-15 lines (285 tokens); a causal line is in the top 15 for 94% of incidents |
| BGL anomaly F1 (time split, 256-line windows) | **0.954** (supervised) vs 0.545 supervised template-count LogReg, 0.335 Drain3+DeepLog |
| HDFS anomaly F1 (per-block sessions) | **0.965** supervised vs 0.969 supervised LogReg — saturated, no real gain |
| Unseen system (Thunderbird) | zero-shot: **fails** (F1 = always-anomalous baseline). With 1% labels (2.1k windows): F1 **0.985** vs 0.858 for LogReg trained on 100% of labels |
| Model size | 33.6M + 3.6M params; int8 ONNX **37.5 MB** total (FP32 148 MB) |
| int8 vs FP32 accuracy | F1 drop <= 0.5 point on BGL / HDFS / lab |
| CPU throughput, int8, template cache (1M raw BGL lines) | **17.7k lines/s pinned to 4 cores** (target 20k: **missed**), 24.1k on 8 cores, 9.9k on 1 core; HDFS 16.8k / 22.7k / 8.3k — [`docs/serving_benchmark.md`](docs/serving_benchmark.md) |
| Frontier-LLM comparison | **not run** (no API key in the build environment); a 1.5B open LLM is included |

Full tables: [`results/benchmark.md`](results/benchmark.md) (regenerate with `make eval`),
[`results/ablations.md`](results/ablations.md), [`docs/error_analysis.md`](docs/error_analysis.md).

## Quickstart

```bash
pip install -e ".[dev,train,serve,baselines]"
make test                                   # 100+ tests
python -m loglens.serve.cli rank sample.log # needs artifacts/onnx (see "Reproduce")
docker build -t loglens . && docker run --rm -v "$PWD:/data" loglens rank /data/sample.log   # untested here: no Docker on the dev machine
```

Demo (simulated fault, no Docker): `python scripts/demo.py --fault db_pool_exhaustion` — LogLens flags the
window and ranks the `QueuePool limit reached` lines at the top.

CLI: `loglens rank <file> [--from ISO --to ISO]`, `loglens watch <paths>`, `loglens bench <file>`.
HTTP: `uvicorn loglens.serve.api:app` -> `POST /score {"lines": [...], "top_k": 15}`, `GET /health`,
`GET /metrics`. Agent tool: `loglens.agent.tool.TOOL_SPEC` / `LogLensTool.rank_suspect_lines`.

## How it works

```
raw lines -> parse (ts, level, service) -> mask (<NUM> <IP> <PATH> <DURATION:bucket> ...)
          -> BPE (16k) -> line encoder (8x512, 33.6M) -> 256-d embedding  --[LRU template cache]
          -> window model (4x256) over 256-line windows
                +-- anomaly head (CLS)
                +-- suspicion head (per line)   -> ranked suspect lines
                +-- template head (unsupervised objective)
```

* **Masking** (`loglens/tokenizer/masking.py`): ordered regex rules, 80+ unit tests, keeps duration
  buckets and small status/error codes. Logs are >90% duplicates after masking (HDFS: 11M lines -> 57
  distinct masked lines), so the encoder embeds each *distinct* line once — the same fact the runtime
  cache exploits.
* **Line encoder**: pre-LN transformer, MLM-pretrained on ~59k distinct masked lines from 14 systems.
* **Window model**: line embedding + log-bucketed time gap + service + level embeddings.
* **Fault lab** (`lab/`): 5-service microservice app with 10 fault types and causal-line labels. Docker
  Compose version included (**not run**); results use the deterministic simulator in `lab/sim`.
* **Serving**: ONNX Runtime dynamic int8 (parity vs PyTorch < 1e-3), embedding cache, parse+mask in a
  process pool pipelined with model inference.

## Reproduce

```bash
make data          # download Loghub (Zenodo), parse, time-split, Parquet          (~1 h, ~15 GB)
make lab-campaign  # simulate 10 campaigns x 18 incidents + causal labels
python -m loglens.data.masked && python -m loglens.data.windows
make tokenizer && python -m loglens.data.pretok
make pretrain      # ~50 min on an RTX 3050
python -m loglens.train.embed
make finetune      # supervised; python -m loglens.train.finetune --config configs/finetune_unsup.yaml
make eval          # results/benchmark.md
make export        # ONNX + int8 -> artifacts/onnx
```

## Limitations (read this)

* **Synthetic incidents.** Root-cause numbers are on a simulator I wrote, whose fault taxonomy the
  model was trained on. They show the pipeline works, not that it will find real production causes.
* **Zero-shot transfer failed.** On the held-out system (Thunderbird, first 6M lines) neither the
  supervised nor the unsupervised model beats the always-anomalous baseline; only few-label
  fine-tuning works. The pretrained encoder is not measurably better than a random-init one on the
  in-distribution tasks (see ablations).
* **In-distribution F1 overstates generalisation**: test lines are mostly templates seen in train
  (`docs/data_report.md`), and HDFS is saturated (a template-count logistic regression matches it).
* **Gates not met**: tokenizer >= 2x fewer tokens than GPT-2 (got 1.5x; 1.3x on the unseen system);
  nearest-neighbour template agreement >= 90% (got 68%); linear probe beating TF-IDF (it did not);
  throughput on 4 pinned cores (17.7k vs 20k lines/s; the Python parse+mask hot path is the limit).
* **No frontier-LLM baseline** was run, so the "within 10% of a frontier LLM at 1/100th of the cost"
  goal is untested. No cost-per-1M-lines table is claimed.
* **Train/serve skew on unknown formats.** The model saw message text only. With a known format
  (`loglens rank --format bgl`) the served model reproduces the offline F1 (0.960 vs 0.954); with the
  generic parser it drops to 0.79 ([`docs/serve_consistency.md`](docs/serve_consistency.md)).
* Thunderbird / Spark are truncated subsets; only English-ish logs were tried; no real customer data.

## Repository map

`loglens/data` (download, parsers, splits, windows) · `loglens/tokenizer` · `loglens/model` ·
`loglens/train` · `loglens/eval` (metrics, baselines, LLM harness, report) · `loglens/serve`
(export, cache, runtime, API, CLI) · `loglens/agent` · `lab/` · `configs/` · `docs/` ·
[`progress.md`](progress.md) (build log and deviations).
