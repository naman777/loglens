# CPU serving benchmark

`loglens bench` on the first 1,000,000 raw lines of BGL and HDFS (generic parser, no dataset specific header handling). Batches of 20,000 lines; the whole process tree is pinned to N cores with `cpu_affinity` (the machine has 16 logical cores; numbers are for the pinned subset). ONNX Runtime 1.30, dynamic int8 unless stated. Peak RSS is the process (workers excluded).

| dataset | configuration | lines/s | cache hit rate | p50 / p99 batch (ms) | peak RSS (MB) |
| --- | --- | --- | --- | --- | --- |
| BGL | int8 + cache, 4 cores (2 ORT threads, 3 mask workers) | 17,703 | 91.8% | 1055 / 2477 | 659 |
| BGL | int8 + cache, 4 cores, single process (4 ORT threads) | 10,009 | 91.8% | 2121 / 3422 | 657 |
| BGL | int8 + cache, 8 cores (4 ORT threads, 6 mask workers) | 24,088 | 91.8% | 808 / 1784 | 661 |
| BGL | int8 + cache, 1 core | 9,864 | 91.8% | 1953 / 3541 | 656 |
| BGL | fp32 + cache, 4 cores (2 ORT threads, 3 mask workers) | 12,091 | 91.8% | 1506 / 3822 | 884 |
| BGL | int8, NO cache, 4 cores (2 ORT threads, 3 mask workers) | 14,652 | 0.0% | 1177 / 3157 | 968 |
| HDFS | int8 + cache, 4 cores (2 ORT threads, 3 mask workers) | 16,778 | 96.4% | 1148 / 2232 | 469 |
| HDFS | int8 + cache, 4 cores, single process (4 ORT threads) | 9,062 | 96.4% | 2073 / 3625 | 469 |
| HDFS | int8 + cache, 8 cores (4 ORT threads, 6 mask workers) | 22,660 | 96.4% | 838 / 1717 | 469 |
| HDFS | int8 + cache, 1 core | 8,290 | 96.4% | 2401 / 3404 | 469 |
| HDFS | fp32 + cache, 4 cores (2 ORT threads, 3 mask workers) | 11,987 | 96.4% | 1614 / 2701 | 514 |
| HDFS | int8, NO cache, 4 cores (2 ORT threads, 3 mask workers) | 15,123 | 0.0% | 1289 / 2335 | 469 |
