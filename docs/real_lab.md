# Real Docker lab validation

The saved model benchmarks use the simulator. The first real Docker CI campaign passed all six
fault/recovery checks in run 36725075577. Evidence review found that container recreation discarded
earlier logs; the runner now captures and merges snapshots before each recreation/recovery.
The corrected campaign is being validated separately. Docker remains absent on the local machine.

## Run

Requires Docker Engine/Desktop with Compose v2 and Linux containers. The runner itself uses only
Python's standard library; building the service images installs their dependencies.

```bash
python lab/run_real_campaign.py --duration 10
# A shorter targeted check:
python lab/run_real_campaign.py --faults db_pool_exhaustion disk_full --duration 2
```

The runner uses an isolated `loglens-lab` Compose project, and refuses to start if that project
already contains containers. Its default cleanup removes only this project's containers and volumes.
`--keep-up` leaves them running for inspection; clean them up afterwards with:

```bash
docker compose -p loglens-lab -f lab/docker-compose.yml down --volumes
```

Gateway access is bound to localhost port 8000. Do not expose debug endpoints to a public network.
Services have CPU/memory limits; the worker spool is a **16 MiB tmpfs**. Disk-full injection verifies
that mount type and size and caps its writes before filling it. The spool is intentionally ephemeral.

## What now actually happens

- Orders insert into Postgres and enqueue Redis tasks. A failed dependency returns HTTP 503.
- The worker consumes those tasks, spools an email marker, and acknowledges only successful writes.
  Failed tasks stay pending and are retried by the single consumer after recovery.
- Payments use a bounded number of real Postgres connections. `/debug/hold` acquires connections;
  concurrent charges fail when capacity is exhausted; `/debug/release` closes held connections.
- Inventory reads Redis and performs a real Postgres query on a cache miss/outage. Its stock response
  remains a fixed lab fixture; there is no production catalogue or external bank integration.
- Gateway propagates upstream HTTP failures instead of returning them as successful responses.
- Invalid-port and slow-bank faults recreate the serving container through a Compose override.
- Commands fail on nonzero exit status; injection contexts attempt recovery on every exit path.

## Supported fault effects

| Fault | Required observable evidence |
| --- | --- |
| container_kill | Orders container stopped and gateway order request fails |
| redis_down | Redis stopped and order enqueue fails through gateway |
| db_pool_exhaustion | Ten connections held, concurrent charge returns 503 |
| disk_full | Bounded tmpfs returns ENOSPC; worker task returns 507 |
| bad_config | Serving orders container exits nonzero; gateway request fails |
| slow_downstream | Payment succeeds after >=2.5 s; gateway's 2 s deadline yields 504 |

Each effect must be observed twice, separated by the requested fault duration. Baseline and recovery
both verify all service health endpoints, gateway browse/payment requests, and an order that travels
through Postgres -> Redis -> worker spool. Background gateway traffic continues during injection.

Network latency, packet loss, CPU hog and memory leak remain **simulator-only**. The previous
unverified Docker helpers were replaced; the real runner refuses unsupported faults.

## Evidence and labels

Each run gets a unique directory under `lab/real-out/`, separate from `lab/out/` and training data:

- `incidents.jsonl`: only incidents with observed effects and successful recovery; stores timestamps,
  target service and before/fault/after evidence (Redis itself is the target for `redis_down`). `causal_labels_reviewed` stays false.
- `traffic.jsonl`: actual gateway response statuses and request durations throughout the campaign.
- `compose.log`: container stdout snapshots, including startup failures; unchanged logs repeat across snapshots.
- `logs.jsonl`: merged application JSON logs, preserving events before containers are recreated.
- `failure.json`: runner error, if any; a failed incident is not appended as a successful label.

Manually review cross-service causal evidence before using these logs for RCA scoring. Successful
fault verification is not a causal-line annotation, and a clean CI run is not a new model benchmark.
The order database write and Redis enqueue are deliberately separate operations; failed enqueue can
leave a persisted order. This lab does not claim transactional outbox or exactly-once delivery.
