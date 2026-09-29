# Tokenizer report

Tokens per line on held-out **test** lines (sampled by line, so frequent templates weigh in). LogLens counts include `[CLS]`/`[SEP]` and the level/service prefix tokens. `Thunderbird` is the unseen system: excluded from BPE training.

| system | GPT-2 raw (mean / p95) | GPT-2 masked (mean) | LogLens 8k (mean / p95 / UNK% / >128%) | LogLens 16k (mean / p95 / UNK% / >128%) | LogLens 8k content-only mean | ratio raw-GPT2 / LogLens-8k content |
| --- | --- | --- | --- | --- | --- | --- |
| Android | 24.4 / 71 | 37.4 | 25.4 / 76 / 0.00 / 0.36 | 23.4 / 66 / 0.00 / 0.14 | 22.4 | 1.09x |
| Apache | 20.4 / 29 | 25.6 | 14.5 / 21 / 0.00 / 0.00 | 14.1 / 18 / 0.00 / 0.00 | 11.5 | 1.78x |
| BGL | 24.5 / 39 | 36.9 | 26.0 / 48 / 0.00 / 0.00 | 24.5 / 44 / 0.00 / 0.00 | 23.0 | 1.07x |
| HDFS | 36.2 / 54 | 34.9 | 17.4 / 27 / 0.00 / 0.00 | 15.9 / 23 / 0.00 / 0.00 | 14.4 | 2.52x |
| HPC | 8.3 / 32 | 23.0 | 11.5 / 30 / 0.00 / 0.01 | 10.6 / 28 / 0.00 / 0.01 | 8.5 | 0.98x |
| Hadoop | 27.4 / 47 | 38.3 | 22.5 / 35 / 0.00 / 0.00 | 20.6 / 30 / 0.00 / 0.00 | 19.5 | 1.40x |
| HealthApp | 37.5 / 52 | 58.7 | 32.0 / 37 / 0.00 / 0.01 | 29.2 / 36 / 0.00 / 0.01 | 29.0 | 1.29x |
| Lab | 14.6 / 21 | 29.3 | 16.6 / 24 / 0.00 / 0.00 | 15.9 / 22 / 0.00 / 0.00 | 13.6 | 1.07x |
| Linux | 12.7 / 20 | 26.4 | 15.4 / 21 / 0.00 / 0.00 | 15.1 / 19 / 0.00 / 0.00 | 12.4 | 1.03x |
| Mac | 32.0 / 78 | 41.2 | 27.3 / 53 / 0.00 / 0.50 | 25.0 / 49 / 0.00 / 0.50 | 24.3 | 1.31x |
| OpenStack | 102.5 / 136 | 49.9 | 29.5 / 41 / 0.00 / 0.00 | 28.2 / 39 / 0.00 / 0.00 | 26.5 | 3.86x |
| Proxifier | 35.4 / 44 | 53.9 | 30.9 / 37 / 0.00 / 0.00 | 30.9 / 37 / 0.00 / 0.00 | 27.9 | 1.27x |
| SSH | 26.0 / 47 | 37.0 | 20.7 / 34 / 0.00 / 0.00 | 20.3 / 34 / 0.00 / 0.00 | 17.7 | 1.47x |
| Spark | 5.8 / 6 | 20.2 | 8.1 / 8 / 0.00 / 0.00 | 8.1 / 8 / 0.00 / 0.00 | 5.1 | 1.13x |
| Thunderbird (unseen) | 43.2 / 112 | 44.5 | 36.2 / 75 / 0.00 / 0.01 | 34.4 / 71 / 0.00 / 0.01 | 33.2 | 1.30x |
| Zookeeper | 38.4 / 76 | 49.8 | 30.7 / 56 / 0.00 / 0.20 | 28.5 / 52 / 0.00 / 0.20 | 27.7 | 1.39x |

Mean ratio across systems: 1.50x; unseen system (Thunderbird): 1.30x fewer content tokens than raw GPT-2 (gate: >= 2x on the unseen system, < 2% of lines over 128 tokens).

Vocabulary choice: mean tokens/line 22.8 (8k) vs 21.5 (16k) -> rule '8k if within 5% of 16k' selects **16k**.
