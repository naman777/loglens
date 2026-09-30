# Real-container validation and diagnosis pilot

Validated in [GitHub Actions run 36725870364](https://github.com/naman777/loglens/actions/runs/36725870364)
on Linux, commit `d96595b`, with six faults and a 2-second minimum sustained-effect interval.
All **6/6** injections showed the required effect twice and passed end-to-end recovery checks.

| Fault | Observed evidence |
| --- | --- |
| container_kill | Orders exited; gateway connection-refused errors and failed requests |
| redis_down | Redis exited; inventory Redis errors and order/worker dependency failures |
| db_pool_exhaustion | Ten real connections held; pool exhaustion logs and HTTP 503 |
| disk_full | Worker ENOSPC logs and HTTP 507; pending work retried after freeing space |
| bad_config | Orders serving container exited nonzero; downstream connection-refused errors |
| slow_downstream | Bank-simulator latency ~3 seconds; gateway 2-second timeout and HTTP 504 |

The campaign retained **604 application log events**, plus traffic records and before/fault/after
checks. Evidence is checked in under `results/real_lab/20260930T140053Z-c36417/` with SHA-256 hashes.
Raw container stdout is also in the CI artifact. An earlier passing run lost pre-recreation logs;
the corrected runner snapshots logs before recreation and recovery and merges repeated snapshots.
Redis is labelled as the failed dependency itself, not as the inventory service that reports it.

These are actual toy-application containers, not the deterministic log simulator. They remain a
small controlled lab, not production incidents. Evidence was inspected by the coding assistant;
**human-reviewed causal-line labels are still absent**. No new Recall@5 accuracy is claimed.

## Actual LLM diagnoses on these six incidents

Qwen2.5-1.5B-Instruct, greedy generation, 1,500 text-token budget plus chat overhead, max 128 output
tokens. Each prompt includes only log timestamp/service/level/message. The time slice is restricted
to the fault interval to avoid contamination from adjacent faults in this short campaign.

| Metric | Chronological raw prefix | LogLens top 15 |
| --- | --- | --- |
| Correct root-cause service | 1/6 (16.7%) | 3/6 (50.0%) |
| Valid structured answers | 5/6 | 6/6 |
| Mean input tokens | 1,367.7 | 920.8 |
| Mean end-to-end latency | 3.09 s | 2.14 s |

LogLens-assisted diagnoses identified database-pool exhaustion, disk full and slow downstream
correctly. Both arms missed container kill, Redis outage and invalid configuration. Some answers
blamed the reporting gateway/orders/inventory service despite evidence naming the failed dependency.
This is a useful failure mode to address, not grounds for a broad accuracy or speed claim.

The six-case result is weaker than the 31/36 synthetic result. Prompt/schema choices, a small LLM,
context truncation and one short campaign limit interpretation. There are no confidence claims,
production claims or interactive-agent results. Full responses: `results/diagnosis/20260930T140425225762Z/`.

```bash
python scripts/diagnosis_eval.py --real-dir results/real_lab/20260930T140053Z-c36417 --max-prompt-tokens 1500 --max-new-tokens 128
```
