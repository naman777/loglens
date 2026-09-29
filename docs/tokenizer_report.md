# Tokenizer report

Tokens per line on held-out **test** lines (sampled by line, so frequent templates weigh in). LogLens counts include `[CLS]`/`[SEP]` and the level/service prefix tokens. `Thunderbird` is the unseen system: excluded from BPE training.

| system | GPT-2 raw (mean / p95) | GPT-2 masked (mean) | LogLens 8k (mean / p95 / UNK% / >128%) | LogLens 16k (mean / p95 / UNK% / >128%) | LogLens 8k content-only mean | ratio raw-GPT2 / LogLens-8k content |
| --- | --- | --- | --- | --- | --- | --- |
| Android | 24.4 / 71 | 37.3 | 25.2 / 76 / 0.00 / 0.36 | 22.5 / 56 / 0.00 / 0.17 | 22.2 | 1.10x |
| Apache | 20.4 / 29 | 25.6 | 14.9 / 21 / 0.00 / 0.00 | 14.3 / 20 / 0.00 / 0.00 | 11.9 | 1.71x |
| BGL | 24.5 / 39 | 36.9 | 26.1 / 47 / 0.00 / 0.15 | 24.5 / 44 / 0.00 / 0.00 | 23.1 | 1.06x |
| HDFS | 36.2 / 54 | 34.9 | 19.0 / 28 / 0.00 / 0.00 | 16.9 / 26 / 0.00 / 0.00 | 16.0 | 2.26x |
| HPC | 8.3 / 32 | 22.9 | 11.8 / 31 / 0.00 / 0.01 | 11.1 / 28 / 0.00 / 0.01 | 8.8 | 0.94x |
| Hadoop | 27.4 / 47 | 38.3 | 24.5 / 44 / 0.00 / 0.00 | 21.1 / 32 / 0.00 / 0.00 | 21.5 | 1.27x |
| HealthApp | 37.5 / 52 | 58.7 | 32.9 / 39 / 0.00 / 0.01 | 30.5 / 36 / 0.00 / 0.01 | 29.9 | 1.26x |
| Lab | 14.6 / 21 | 29.3 | 16.9 / 23 / 0.00 / 0.00 | 16.0 / 22 / 0.00 / 0.00 | 13.9 | 1.05x |
| Linux | 12.7 / 20 | 26.4 | 16.3 / 22 / 0.00 / 0.00 | 15.2 / 20 / 0.00 / 0.00 | 13.3 | 0.96x |
| Mac | 32.0 / 78 | 41.0 | 28.9 / 53 / 0.00 / 0.53 | 25.9 / 49 / 0.00 / 0.50 | 25.9 | 1.23x |
| OpenStack | 102.5 / 136 | 49.9 | 29.6 / 40 / 0.00 / 0.00 | 28.5 / 39 / 0.00 / 0.00 | 26.6 | 3.85x |
| Proxifier | 35.4 / 44 | 53.9 | 31.1 / 37 / 0.00 / 0.00 | 30.9 / 37 / 0.00 / 0.00 | 28.1 | 1.26x |
| SSH | 26.0 / 47 | 37.0 | 20.9 / 34 / 0.00 / 0.00 | 20.3 / 34 / 0.00 / 0.00 | 17.9 | 1.45x |
| Spark | 5.8 / 6 | 20.2 | 8.1 / 8 / 0.00 / 0.00 | 8.1 / 8 / 0.00 / 0.00 | 5.1 | 1.13x |
| Thunderbird (unseen) | 43.2 / 112 | 44.2 | 37.2 / 80 / 0.00 / 0.01 | 34.5 / 72 / 0.00 / 0.01 | 34.2 | 1.26x |
| Zookeeper | 38.4 / 76 | 49.7 | 31.2 / 56 / 0.00 / 0.20 | 29.0 / 53 / 0.00 / 0.20 | 28.2 | 1.36x |

Mean ratio across systems: 1.45x; unseen system (Thunderbird): 1.26x fewer content tokens than raw GPT-2 (gate: >= 2x on the unseen system, < 2% of lines over 128 tokens).

Vocabulary choice: mean tokens/line 23.4 (8k) vs 21.8 (16k) -> rule '8k if within 5% of 16k' selects **16k**.
