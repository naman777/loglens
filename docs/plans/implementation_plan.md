# LogLens — Implementation Plan

Sep 30, 2026 · @Naman Kundra

## How to use this doc

Work top to bottom: each phase lists numbered steps, then a **Done when** check that must pass before you move on. The Build Plan holds the why, the schedule, and the risks; this doc holds the how.

**Tech stack**

| Layer | Choice |
| --- | --- |
| Language | Python 3.11 (+ small Go/Rust later only if profiling demands it) |
| Data | Polars, PyArrow, Parquet |
| Tokenizer | Hugging Face `tokenizers` (BPE) + your own regex masking |
| Modeling | PyTorch 2.x, plain training loop, `torch.compile` optional |
| Tracking | Weights & Biases (free) or MLflow |
| Baselines | `drain3`, your own DeepLog (LSTM) reimplementation |
| Serving | ONNX Runtime (int8), FastAPI, Typer CLI |
| Lab | Docker Compose, Locust, Pumba / `tc netem`, Postgres, Redis |
| Quality | pytest, ruff, mypy (light), GitHub Actions |

**Repo layout**

```
loglens/
  pyproject.toml
  Makefile
  configs/            # yaml: data, tokenizer, model, train, eval
  loglens/
    data/             # download, parse, normalize, split
    tokenizer/        # masking rules, BPE training, encode()
    model/            # line_encoder.py, window_model.py, heads.py
    train/            # pretrain.py, finetune.py, losses.py
    eval/             # metrics.py, baselines/, llm_eval.py, report.py
    serve/            # export_onnx.py, runtime.py, cache.py, api.py, cli.py
  lab/                # docker-compose.yml, services/, faults/, load/
  scripts/            # one-off runs
  tests/
  notebooks/          # exploration only, never imported
  docs/               # model card, blog drafts
```

## Phase 0 — Environment and repo setup

Goal: a repo where every later step is one `make` command and every run is logged. Budget 2–3 days.

1. Create the GitHub repo `loglens` (public from day one; commit history is part of the story).
2. Set up the environment with `uv` or `conda`; pin Python 3.11, PyTorch, `tokenizers`, polars, pyarrow, onnxruntime, fastapi, typer, drain3, wandb, pytest, ruff.
3. Create the package skeleton exactly as in the repo layout above, each module with an empty `__init__.py`.
4. Add a `Makefile` with targets: `setup`, `data`, `lab-up`, `lab-down`, `tokenizer`, `pretrain`, `finetune`, `eval`, `export`, `serve`, `test`, `lint`.
5. Add config loading: one YAML per stage under `configs/`, loaded into dataclasses; every script takes `--config`.
6. Add a global seed utility (`random`, `numpy`, `torch`, CUDA) and call it at the top of every entry point.
7. Create a W&B project `loglens`; log config, git commit hash, and dataset manifest hash on every run.
8. Add GitHub Actions: ruff + pytest on push.
9. Write a dummy training script that trains a 2-layer MLP on random data and logs loss to W&B.

**Done when:** CI is green, `make test` passes locally, and the dummy run shows a loss curve in W&B tagged with the commit hash.

## Phase 1 — Data acquisition, normalization, splits

Goal: every log source in one Parquet schema, split by time with no leakage. Budget about 1 week.

**1.1 Download**

1. Read the Loghub and Loghub-2.0 READMEs on GitHub (`logpai/loghub`, `logpai/loghub-2.0`) and note each dataset's size, labels, and license.
2. Write `loglens/data/download.py` that fetches each archive, verifies a checksum, and extracts to `data/raw/<system>/`.
3. Start with: HDFS, BGL, Thunderbird (first \~20M lines only), OpenStack, Hadoop, Spark, Zookeeper, Linux, Apache, HPC. Add more later if pretraining needs tokens.

**1.2 Parse into one schema**

1. Define the schema: `system, ts (int64 epoch ms), host, component, level, message, session_id (nullable), label (nullable int8), line_no`.
2. Write one small parser per system in `loglens/data/parsers/` (regex for the header, the rest is `message`).
3. Attach labels: BGL and Thunderbird mark alerts per line (the first field is `-` for normal); HDFS labels are per block ID, so extract `blk_` IDs into `session_id` and join the anomaly label file.
4. Write to `data/parquet/<system>/part-*.parquet`, \~1M rows per shard.

**1.3 Clean**

1. Drop empty and non-UTF-8 lines (log the counts).
2. Truncate messages longer than 1,024 characters and record how many were cut.
3. Keep duplicates in the data but record duplicate rates; dedup is applied later, at split time.

**1.4 Split by time**

1. For each system, sort by `ts`; train = first 70%, validation = next 10%, test = last 20%.
2. For HDFS, split by the first timestamp of each block session so no session straddles two splits.
3. Leakage check: compute the fraction of test messages whose exact masked text appears in train. Report it; do not hide it.
4. Pick one system as **unseen** (e.g. Thunderbird or Spark) and exclude it from pretraining entirely.
5. Save `data/splits/manifest.json` with row counts, time ranges, and a hash.

**1.5 Stats report**

1. Notebook or script that prints per system: lines, time span, anomaly rate, unique templates (via Drain3), average line length.
2. Save it as `docs/data_report.md`; it goes into the blog post later.

**Done when:** all chosen systems are in Parquet, the manifest exists, train and test time ranges do not overlap for any system, and the stats report is written.

## Phase 2 — Fault-injection lab

Goal: logs with ground-truth root causes, produced by a script you can rerun. Timebox to 8 working days, in parallel with Phases 1 and 3.

**2.1 Build a small app you control**

Start with your own 5-service app rather than a large demo; you need to know exactly what each service logs. (The OpenTelemetry demo is a good v2 upgrade.)

1. `gateway` (FastAPI): routes requests.
2. `orders` (FastAPI): writes to Postgres.
3. `payments` (FastAPI): calls a fake bank with random latency, uses a DB connection pool.
4. `inventory` (FastAPI): reads through Redis cache.
5. `worker` (Python): consumes a Redis queue, sends fake emails.
6. Every service logs realistic lines at INFO/WARN/ERROR, including retries, timeouts, and stack traces. Write logs as JSON to stdout with a `service` field.
7. `lab/docker-compose.yml` wires them up with Postgres and Redis; logs collected with `docker compose logs --timestamps` or a Fluent Bit sidecar into `lab/out/<run_id>/logs.jsonl`.

**2.2 Load generator**

1. Locust script with a realistic mix: browse 60%, order 25%, pay 15%.
2. Vary load in waves (low, normal, peak) so "normal" is not a flat line.

**2.3 Fault injector**

1. Write `lab/faults/inject.py` with one function per fault type:
   1. Kill a container (`docker kill`), then restart it.
   2. Network latency on one service (`pumba netem delay` or `tc netem`).
   3. Packet loss between two services.
   4. DB connection-pool exhaustion (hold connections open).
   5. Redis down (cache miss storm).
   6. Disk full on the worker (fill a volume).
   7. Bad config deploy (wrong env var, restart the service).
   8. Memory leak (a debug endpoint that allocates until OOM).
   9. CPU hog (a stress process in one container).
   10. Slow downstream dependency (fake bank latency x10).
2. Each injection writes a label record: `run_id, incident_id, fault_type, target_service, t_start, t_end, notes`.

**2.4 Run campaigns**

1. `lab/run_campaign.py`: start the lab, warm up 5 minutes, then loop: 10–20 min normal, inject a random fault for 2–5 min, recover.
2. Run overnight campaigns; target 150+ incidents in total across all fault types.

**2.5 Mark causal lines**

1. For each incident, auto-label a line as **causal** if it comes from `target_service`, falls within `[t_start, t_end + 60s]`, and is WARN or above or matches the fault's keywords (e.g. `timeout`, `pool`, `connection refused`).
2. Hand-check 30 incidents and fix the rules until auto-labels agree with you on ≥ 90% of lines.
3. Save `data/lab/incidents.parquet` (windows) and `data/lab/causal_lines.parquet`.
4. Split incidents by campaign (not randomly): earlier campaigns for train/val, the last campaign for test.

**Done when:** at least 150 labeled incidents across at least 8 fault types, causal lines marked and hand-checked, and one command (`make lab-campaign`) reproduces a campaign.

## Phase 3 — Log tokenizer

Goal: a tokenizer that understands log structure and uses at least 2× fewer tokens per line than GPT-2's. Budget 4–5 days.

**3.1 Masking rules** (`loglens/tokenizer/masking.py`)

1. Write ordered regex rules; order matters (UUID before HEX before NUM):
   - `<TS>` timestamps inside the message
   - `<UUID>`, `<IP>` (v4 and v6, with optional `:port` → `<IP>:<PORT>`)
   - `<PATH>` file paths and URLs (keep the last path segment as a hint if short)
   - `<HEX>` 0x-prefixed or 8+ hex characters
   - `<BLK>` HDFS block IDs, `<ID>` long alphanumeric IDs
   - `<DURATION:bucket>` for values with ms/s units, buckets `<10ms`, `<100ms`, `<1s`, `<10s`, `10s+`
   - `<NUM>` remaining numbers, but keep HTTP status codes and small error codes (0–999) literal
2. Add structural prefix tokens from the schema: `<LVL:ERROR>`, `<SVC:payments>` (map unseen services to `<SVC:other>`).
3. Unit-test each rule with 5+ positive and negative examples; include tricky cases like version strings (`v2.3.1`) and dates.

**3.2 Train BPE**

1. Sample \~5M masked lines from the pretraining systems, stratified so no system is more than 30% of the sample.
2. Train a byte-level BPE with the `tokenizers` library, vocab 8k and 16k (train both), special tokens: `[PAD] [CLS] [SEP] [MASK] [UNK]` plus all mask tokens.
3. Save to `artifacts/tokenizer/loglens-bpe-{8k,16k}.json`.

**3.3 Evaluate**

1. On held-out test lines (including the unseen system), measure mean and p95 tokens per line for: raw GPT-2 tokenizer, GPT-2 on masked text, LogLens 8k, LogLens 16k.
2. Measure `[UNK]` rate and the share of lines longer than 128 tokens.
3. Pick the vocab size (8k if within 5% of 16k; smaller embeddings help CPU).
4. Write the comparison table into `docs/tokenizer_report.md`.

**Done when:** masking tests pass, the chosen tokenizer gives ≥ 2× fewer tokens per line than raw GPT-2 on the unseen system, and fewer than 2% of lines exceed 128 tokens.

## Phase 4 — Line encoder pretraining

Goal: a \~25–30M parameter encoder whose line embeddings group lines by meaning. Budget 2 weeks, most of it waiting on runs and fixing bugs.

**4.1 Model code** (`loglens/model/line_encoder.py`)

1. Pre-LayerNorm transformer encoder: 8 layers, d\_model 512, 8 heads, FFN 2048, GELU, dropout 0.1, max length 128, learned positions (or RoPE if you want the extra learning).
2. Token embedding tied to the MLM output layer.
3. Pooling: mean over non-pad tokens, then a linear projection to a 256-d line embedding (this is what the window model consumes).
4. Rough size: 8 layers × \~3.1M + 8k × 512 embeddings ≈ 29M parameters. Print the exact count in a test.

**4.2 Data loader**

1. Stream pretraining lines from Parquet (train split only, unseen system excluded), mask → tokenize on the fly or pre-tokenize to `.npy` shards (faster; do this).
2. Bucket by length and pad per batch to cut wasted compute.
3. MLM masking: 15% of tokens; of those 80% `[MASK]`, 10% random, 10% unchanged. Never mask `[CLS]`/`[PAD]`; do mask structural tokens sometimes so the model learns them.

**4.3 Training loop** (`loglens/train/pretrain.py`)

1. AdamW, lr 5e-4, betas (0.9, 0.98), weight decay 0.01, warmup 2k steps, cosine decay to 10%.
2. Batch \~256 lines × up to 128 tokens; bf16 autocast; grad clip 1.0.
3. Target 0.5–1B training tokens total. Log loss, lr, grad norm, tokens/s every 50 steps; checkpoint every 30 minutes and keep the last 3.
4. Optional second objective: SimCSE-style contrastive loss (same line, two dropout masks) with weight 0.1. Try it only after plain MLM works.

**4.4 Smoke test locally first**

1. Run on 1% of data with a 2-layer model on your laptop GPU/CPU until loss clearly drops.
2. Overfit test: a single batch should reach near-zero loss in a few hundred steps. If not, there is a bug.

**4.5 Full run on rented GPU**

1. Rent one RTX 4090 or A100 (Vast.ai / RunPod); copy the pre-tokenized shards; run in `tmux` with W&B logging.
2. Expect hours up to about a day depending on GPU and token count. Stop when validation MLM loss has flattened for 10% of steps.
3. Download the final checkpoint and tokenizer; shut the instance down immediately.

**4.6 Sanity checks**

1. Embed 50k validation lines; for 50 random lines, print 5 nearest neighbours. They should share templates or meaning.
2. Run a UMAP plot coloured by system and by level; errors should cluster separately from INFO noise.
3. Linear probe: freeze the encoder, train logistic regression on BGL line labels. It should beat a TF-IDF + logistic regression baseline.

**Done when:** validation MLM loss has plateaued, nearest-neighbour checks look right in ≥ 90% of spot checks, and the linear probe beats TF-IDF.

## Phase 5 — Window model, heads, fine-tuning

Goal: one model that scores a window for anomaly and scores every line for suspicion. Budget 1 week.

**5.1 Precompute line embeddings**

1. Freeze the line encoder and embed every line in all splits once; save as float16 `.npy` shards keyed by `(system, line_no)`. Window training then runs in minutes, even on CPU.

**5.2 Window model** (`loglens/model/window_model.py`)

1. Input per line: 256-d line embedding + time-gap embedding (log-bucketed seconds since previous line, \~16 buckets) + service embedding + level embedding.
2. Transformer encoder: 4 layers, d\_model 256, 4 heads, window 256 lines, a `[CLS]` token at the front. About 3–5M parameters.
3. Windows: sliding by time (e.g. 60 s) or by count (256 lines, stride 128) — try both, keep the one that validates better.

**5.3 Unsupervised pretraining (no labels)**

1. Mask 15% of line positions; predict each masked line's template ID (from Drain3 clusters over train, top 2,000 templates + `other`).
2. Train on normal-only windows (label 0 or unlabeled) from all pretraining systems plus lab normal periods.
3. Unsupervised anomaly score = mean negative log-likelihood of the true template over masked positions, averaged across several random masks at inference.

**5.4 Heads** (`loglens/model/heads.py`)

1. Anomaly head: MLP on `[CLS]` → 1 logit.
2. Suspicion head: MLP on each line position → 1 logit per line.

**5.5 Supervised fine-tuning** (`loglens/train/finetune.py`)

1. Anomaly: BCE with positive-class weight = negatives / positives on the train split; data from BGL, Thunderbird, HDFS sessions, and lab incidents.
2. Suspicion: on lab incident windows only, BCE on causal-line labels + a pairwise margin ranking loss (causal lines above non-causal) with weight 0.5.
3. Stage A: encoder frozen, train window model + heads. Stage B (optional): unfreeze the top 2 encoder layers with lr 1e-5 and recompute embeddings on the fly.
4. Early stopping on validation PR-AUC (anomaly) and Recall@5 (suspicion).
5. Tune the anomaly threshold on validation only; freeze it before touching test.

**5.6 Ablations to run** (these become blog material)

1. With vs without the time-gap embedding.
2. Masking level: full masking vs keeping bucketed durations.
3. Pretrained vs randomly initialised line encoder.
4. Unsupervised only vs supervised.

**Done when:** the model beats Drain + DeepLog on BGL F1 (time split) and reaches RCA Recall@5 ≥ 0.6 on lab validation incidents.

## Phase 6 — Baselines and evaluation harness

Goal: a benchmark table where every number comes from one reproducible command. Budget 1 week.

**6.1 Shared harness** (`loglens/eval/`)

1. One `Evaluator` interface: `fit(train_windows)`, `score_windows(windows) -> anomaly scores`, `rank_lines(window) -> per-line scores` (optional).
2. All methods see exactly the same windows and splits, loaded from the manifest.
3. Metrics module (see Appendix): precision, recall, F1 at the validation-tuned threshold, PR-AUC, Recall@1/@5, MRR, latency, throughput.
4. `make eval` writes `results/results.json` and renders `results/benchmark.md`.

**6.2 Classic baselines**

1. **Drain3 + DeepLog:** parse templates with Drain3 on train; reimplement DeepLog (2-layer LSTM, next-template prediction, anomaly if the true next template is not in top-k).
2. **Drain3 + frequency baseline:** isolation forest on template count vectors per window. Cheap and often surprisingly strong; include it.
3. **LogBERT-style:** only if time allows; otherwise cite published numbers clearly marked as not reproduced.

**6.3 LLM baselines**

1. **Small open LLM (1–3B instruct), zero-shot, on CPU or free GPU:** prompt with the masked window, ask for an anomaly yes/no + confidence and the IDs of the top 5 suspect lines, in JSON.
2. **Frontier LLM API:** same prompt, same JSON schema, on a fixed random sample of 300–500 test windows (plus all lab test incidents).
3. Cache every response to disk keyed by a hash of the prompt so reruns are free.
4. Record tokens in/out per call; cost = tokens × the published price on the date you ran it. Put the date and price in the table footnote.
5. Truncate windows that do not fit the context and record how many were truncated (this is part of the argument).

**6.4 Experiments to report**

1. In-distribution anomaly detection: BGL, Thunderbird, HDFS (time split).
2. Unseen-system transfer: the held-out system, zero-shot and with 1% labels.
3. Root-cause ranking: lab test incidents, Recall@1/@5 and MRR, per fault type.
4. Efficiency: lines/s and p50/p99 window latency on the same 4-core CPU for every non-API method.

**6.5 Error analysis**

1. Pull the 30 worst misses per experiment and label why (masking hid the signal, window too short, rare template, bad label).
2. Write 5–10 bullet findings; they become the "what didn't work" section of the blog.

**Done when:** every cell of the benchmark table is filled by `make eval` with a fixed seed, and the table and error analysis are committed.

## Phase 7 — CPU serving

Goal: int8 inference on a 4-core laptop at ≥ 20k lines/s with the template cache, behind a CLI and an HTTP API. Budget 1 week.

**7.1 Export** (`loglens/serve/export_onnx.py`)

1. Export the line encoder and the window model + heads as two ONNX graphs with dynamic batch and sequence axes.
2. Verify: run the same 1,000 inputs through PyTorch and ONNX; max absolute difference < 1e-3.

**7.2 Quantize**

1. ONNX Runtime dynamic int8 quantization (`quantize_dynamic`, weights int8) on both graphs.
2. If accuracy drops more than 2 F1 points, try static quantization with a calibration set of 1,000 validation windows, or keep the window model FP32 (it is small).
3. Re-run `make eval` on the int8 models and add them as their own rows.

**7.3 Template cache** (`loglens/serve/cache.py`)

1. Key = hash of the masked line text; value = its 256-d embedding.
2. LRU with a size cap (e.g. 200k entries ≈ 100 MB in float16).
3. Report the hit rate on each test system; this number explains most of the speedup.

**7.4 Runtime** (`loglens/serve/runtime.py`)

1. Streaming reader: `tail -f` style for files, or stdin; parse → mask → cache lookup → batch misses (up to 512 lines or 50 ms) → encoder.
2. Maintain a rolling window per service; score the window every N seconds or N lines.
3. Set ONNX Runtime intra-op threads to the physical core count; benchmark 1, 2, 4 threads.

**7.5 Interfaces**

1. CLI with Typer: `loglens watch <paths>` (live), `loglens rank <file> --from --to` (one-off), `loglens bench <file>`.
2. HTTP API with FastAPI: `POST /score` (lines in, anomaly score + ranked lines out), `GET /health`, `GET /metrics` (lines/s, cache hit rate).
3. Output JSON for machines, a coloured table for humans.

**7.6 Package and benchmark**

1. Dockerfile on a slim Python base with only onnxruntime + deps (no PyTorch); target image under 500 MB.
2. `loglens bench` on a 1M-line file: report lines/s, p50/p99 latency, peak RAM, cache hit rate, for FP32 vs int8 and cache on vs off.

**Done when:** int8 accuracy is within 2 F1 points of FP32, throughput is ≥ 20k lines/s on a 4-core CPU with cache, and `docker run loglens rank sample.log` works on a clean machine.

## Phase 8 — Agent integration, demo, docs, release

Goal: LogLens working inside the on-call agent, and a release a stranger can reproduce in 15 minutes. Budget about 2 weeks.

**8.1 Agent tool**

1. Expose one tool to the agent: `rank_suspect_lines(service: str, window: str, top_k: int = 15)` → anomaly score, anomaly timeline, top-k lines with scores and timestamps.
2. The agent calls it first on every alert, then reasons only over the returned lines.
3. Measure on 30 lab incidents, agent with vs without LogLens: tokens sent to the LLM per incident, time to answer, and whether the agent named the correct root-cause service.

**8.2 Live demo**

1. Script: start the lab → normal traffic → run `loglens watch` in a split terminal → inject DB pool exhaustion → LogLens flags it and ranks the pool timeout lines at the top → the agent writes its summary.
2. Record a 2-minute screen capture with voice-over; put the headline number on screen in the first 10 seconds.

**8.3 Model release**

1. Push FP32 + int8 weights and the tokenizer to Hugging Face.
2. Model card: intended use, training data (with Loghub licenses), splits, metrics table, limitations (English-ish logs, tested systems only, synthetic lab incidents), carbon/compute used.

**8.4 Repo polish**

1. README order: one-line pitch → demo GIF → benchmark table → quickstart (3 commands) → architecture diagram → how to reproduce → limitations.
2. Tag `v1.0.0`; make sure `make data && make eval` reproduces the headline number from a fresh clone.
3. Ask one friend to follow the quickstart cold and time it.

**8.5 Write-up and outreach**

1. Blog post (\~2,000 words): the problem, tokenizer design, two-level model, fault lab, results, what failed, cost analysis.
2. Post on LinkedIn/X and relevant subreddits with the demo GIF; submit to Hacker News as a Show HN.
3. Resume bullet: what you built, the headline number, the stack.
4. Short cold email to AI SRE / observability startups linking the demo and the table.

**Done when:** v1.0.0 is tagged, the model is on Hugging Face, the blog and video are live, and a friend reproduced the demo from the README in under 15 minutes.

## Appendix

**A. Default hyperparameters** (starting points; tune on validation only)

| Component | Setting | Default |
| --- | --- | --- |
| Tokenizer | Vocab size | 8,000 |
| Tokenizer | Max tokens per line | 128 |
| Line encoder | Layers / d\_model / heads / FFN | 8 / 512 / 8 / 2048 |
| Line encoder | Line embedding size | 256 |
| Pretraining | Optimizer | AdamW, lr 5e-4, wd 0.01, betas (0.9, 0.98) |
| Pretraining | Schedule | 2k warmup, cosine to 10% |
| Pretraining | Batch | 256 lines, bf16, grad clip 1.0 |
| Pretraining | Tokens | 0.5–1B |
| Window model | Layers / d\_model / heads | 4 / 256 / 4 |
| Window model | Window / stride | 256 lines / 128 |
| Fine-tuning | lr (heads) / lr (unfrozen encoder) | 3e-4 / 1e-5 |
| Fine-tuning | Ranking loss weight | 0.5 |
| Serving | Cache size / micro-batch | 200k entries / 512 lines or 50 ms |

**B. Metric definitions**

- **Precision / Recall / F1:** window-level, at the threshold chosen on validation.
- **PR-AUC:** area under the precision–recall curve; preferred over ROC-AUC because anomalies are rare.
- **Recall@k (RCA):** fraction of incidents where at least one causal line is in the top k ranked lines.
- **MRR:** mean of 1 / rank of the first causal line across incidents.
- **Throughput:** lines processed per second end to end (parse → score), single process, stated core count.
- **Cost per 1M lines:** API tokens × published price for LLMs; for LogLens, state "CPU only, no API cost" plus hardware used.

**C. Debugging checklist**

- Loss not dropping → overfit one batch first; check masking is not masking `[PAD]`; check lr warmup.
- Loss NaN → lower lr, confirm grad clipping, check for empty sequences after masking.
- Great validation, poor test → look for leakage (random split, duplicated templates across splits) or threshold tuned on test.
- Suspicion head ranks everything equally → check causal labels are aligned to the right window positions; increase ranking loss weight.
- int8 accuracy collapse → keep LayerNorm and the heads in FP32; try static quantization with calibration.
- CPU slower than expected → check the cache hit rate, thread count vs physical cores, and batch size.
- Lab incidents look identical → vary load, fault duration, and target service; add noise faults that do not cause incidents.
