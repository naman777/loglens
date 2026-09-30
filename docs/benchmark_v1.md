# Benchmark

Fixed seed; all numbers produced by `make eval`. Anomaly metrics are window-level at a threshold tuned on the validation split only (test never touched). Lab incidents are **synthetic** (simulated microservice lab).

## Anomaly detection: BGL

Test windows: 7,364 (10.9% positive).

| method | precision | recall | F1 | PR-AUC |
| --- | --- | --- | --- | --- |
| always-anomalous | 0.109 | 1.000 | 0.196 | 0.109 |
| Drain3+IsolationForest | 0.556 | 0.094 | 0.161 | 0.282 |
| Drain3+DeepLog | 0.202 | 0.984 | 0.335 | 0.257 |
| LogLens-unsup | 0.101 | 0.118 | 0.109 | 0.157 |
| LogLens-sup | 0.993 | 0.917 | 0.954 | 0.961 |

## Anomaly detection: HDFS

Test windows: 40,000 (2.1% positive).

| method | precision | recall | F1 | PR-AUC |
| --- | --- | --- | --- | --- |
| always-anomalous | 0.021 | 1.000 | 0.040 | 0.021 |
| Drain3+IsolationForest | 0.024 | 0.593 | 0.046 | 0.035 |
| Drain3+DeepLog | 0.674 | 0.597 | 0.633 | 0.449 |
| LogLens-unsup | 0.532 | 0.524 | 0.528 | 0.542 |
| LogLens-sup | 0.934 | 0.999 | 0.965 | 0.999 |

## Anomaly detection: Thunderbird (unseen system)

Test windows: 40,000 (58.2% positive).

| method | precision | recall | F1 | PR-AUC |
| --- | --- | --- | --- | --- |
| always-anomalous | 0.582 | 1.000 | 0.736 | 0.582 |
| Drain3+IsolationForest | 0.677 | 0.867 | 0.761 | 0.585 |
| Drain3+DeepLog | 0.805 | 0.809 | 0.807 | 0.782 |
| LogLens-unsup | 0.582 | 1.000 | 0.736 | 0.396 |
| LogLens-sup | 0.593 | 0.983 | 0.740 | 0.528 |

LogLens-sup fine-tuned on 2100 labelled Thunderbird windows (1%): F1 0.985 ± 0.004, PR-AUC 0.997 (3 seeds).

## Root-cause ranking (lab test incidents)

| method | Recall@1 | Recall@5 | MRR | incidents |
| --- | --- | --- | --- | --- |
| random | 0.083 | 0.167 | 0.137 | 36 |
| severity-heuristic | 0.028 | 0.194 | 0.108 | 36 |
| Drain3+DeepLog (surprise) | 0.528 | 0.806 | 0.653 | 36 |
| LogLens-sup | 0.611 | 0.944 | 0.760 | 36 |

Per fault type, Recall@5:

| method | bad_config | container_kill | cpu_hog | db_pool_exhaustion | disk_full | memory_leak | network_latency | packet_loss | redis_down | slow_downstream |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0.250 | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 | 0.667 | 0.000 |
| severity-heuristic | 0.000 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 | 0.500 | 0.250 | 0.000 | 1.000 |
| Drain3+DeepLog (surprise) | 1.000 | 0.750 | 0.667 | 1.000 | 1.000 | 1.000 | 0.000 | 0.750 | 1.000 | 1.000 |
| LogLens-sup | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
