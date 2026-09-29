# Tokenizer report

Tokens per line on held-out **test** lines (sampled by line, so frequent templates weigh in). LogLens counts include `[CLS]`/`[SEP]` and the level/service prefix tokens. `Thunderbird` is the unseen system: excluded from BPE training.

| system | GPT-2 raw (mean / p95) | GPT-2 masked (mean) | LogLens 8k (mean / p95 / UNK% / >128%) | LogLens 16k (mean / p95 / UNK% / >128%) | LogLens 8k content-only mean | ratio raw-GPT2 / LogLens-8k content |
| --- | --- | --- | --- | --- | --- | --- |
| Apache | 20.4 / 29 | 25.6 | 14.3 / 21 / 0.00 / 0.00 | 14.2 / 21 / 0.00 / 0.00 | 11.3 | 1.81x |
| BGL | 24.5 / 39 | 36.7 | 25.8 / 48 / 0.00 / 0.13 | 25.5 / 47 / 0.00 / 0.13 | 22.8 | 1.08x |
| HDFS | 36.2 / 54 | 34.8 | 16.1 / 24 / 0.00 / 0.00 | 15.7 / 23 / 0.00 / 0.00 | 13.1 | 2.77x |
| HPC | 8.3 / 32 | 22.9 | 10.9 / 28 / 0.00 / 0.00 | 10.7 / 28 / 0.00 / 0.00 | 7.9 | 1.05x |
| Hadoop | 27.4 / 47 | 38.3 | 20.7 / 29 / 0.00 / 0.00 | 20.3 / 29 / 0.00 / 0.00 | 17.7 | 1.55x |
| Lab | 14.6 / 21 | 29.3 | 16.0 / 22 / 0.00 / 0.00 | 15.8 / 21 / 0.00 / 0.00 | 13.0 | 1.12x |
| Linux | 12.7 / 20 | 26.4 | 15.2 / 20 / 0.00 / 0.00 | 15.1 / 20 / 0.00 / 0.00 | 12.2 | 1.04x |
| OpenStack | 102.5 / 136 | 49.8 | 28.5 / 39 / 0.00 / 0.00 | 28.2 / 39 / 0.00 / 0.00 | 25.5 | 4.02x |
| SSH | 26.0 / 47 | 37.0 | 20.0 / 34 / 0.00 / 0.00 | 20.0 / 34 / 0.00 / 0.00 | 17.0 | 1.53x |
| Spark | 5.8 / 6 | 20.2 | 9.1 / 9 / 0.00 / 0.00 | 9.1 / 9 / 0.00 / 0.00 | 6.1 | 0.95x |
| Thunderbird (unseen) | 43.2 / 112 | 44.1 | 37.7 / 77 / 0.00 / 0.01 | 37.7 / 77 / 0.00 / 0.01 | 34.7 | 1.24x |
| Zookeeper | 38.4 / 76 | 49.7 | 28.7 / 51 / 0.00 / 0.20 | 28.1 / 49 / 0.00 / 0.20 | 25.7 | 1.49x |

Mean ratio across systems: 1.64x; unseen system (Thunderbird): 1.24x fewer content tokens than raw GPT-2 (gate: >= 2x on the unseen system, < 2% of lines over 128 tokens).
