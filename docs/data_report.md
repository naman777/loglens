# Data report

Split manifest hash `f4e1693b93d6`. Time-based 70/10/20 splits; sessions are assigned by first timestamp. Unseen (held-out) system: **Thunderbird** (excluded from tokenizer/encoder/window pretraining).

| system | lines | time span | anomaly rate | distinct masked lines | Drain3 clusters | avg chars | dup rate (raw) | truncated | dropped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Android | 1,487,190 | 2017-12-17 → 2017-12-18 (24 h) | - | 69,458 | 10355 | 75 | 82.0% | 687 | 67,815 |
| Apache | 56,482 | 2005-06-09 → 2006-02-28 (6334 h) | - | 58 | 30 | 57 | 74.9% | 0 | 0 |
| BGL | 4,713,493 | 2005-06-03 → 2006-01-04 (5152 h) | 7.39% | 70,186 | 429 | 31 | 92.4% | 0 | 34,470 |
| HDFS | 11,175,629 | 2008-11-09 → 2008-11-11 (39 h) | 2.58% | 57 | 46 | 90 | 7.5% | 0 | 0 |
| HPC | 433,490 | 2003-12-26 → 2006-04-30 (20543 h) | - | 1,194 | 138 | 36 | 97.6% | 0 | 0 |
| Hadoop | 393,433 | 2015-10-17 → 2015-10-19 (51 h) | 93.57% | 2,811 | 374 | 76 | 84.1% | 0 | 877 |
| HealthApp | 253,395 | 2017-12-23 → 2017-12-31 (194 h) | - | 701 | 333 | 69 | 17.5% | 0 | 0 |
| Linux | 25,454 | 2005-01-01 → 2005-12-31 (8736 h) | - | 1,007 | 413 | 56 | 55.7% | 0 | 113 |
| Mac | 116,735 | 2017-07-01 → 2017-07-08 (168 h) | - | 3,037 | 853 | 92 | 60.6% | 77 | 548 |
| OpenStack | 207,820 | 2017-05-14 → 2017-05-17 (64 h) | 8.87% | 445 | 45 | 196 | 27.1% | 0 | 0 |
| Proxifier | 21,329 | 2015-07-26 → 2015-10-30 (2312 h) | - | 1,877 | 77 | 99 | 51.6% | 0 | 0 |
| SSH | 655,147 | 2015-01-01 → 2015-12-31 (8760 h) | - | 5,666 | 46 | 75 | 72.1% | 0 | 0 |
| Spark | 5,998,826 | 2015-09-01 → 2017-04-14 (14167 h) | - | 3,335 | 329 | 54 | 64.0% | 0 | 1,174 |
| Thunderbird (unseen) | 6,000,000 | 2005-11-09 → 2005-11-20 (271 h) | 5.02% | 29,507 | 1034 | 50 | 92.2% | 0 | 0 |
| Zookeeper | 74,380 | 2015-07-29 → 2015-08-25 (642 h) | - | 136 | 72 | 55 | 61.0% | 0 | 0 |

## Leakage check

Fraction of **test** lines whose exact masked text (level + service + masked message) also appears in the train split. This is reported, not hidden: high values mean a model can score well by template lookup, so anomaly F1 on those systems overstates generalisation.

| system | test lines seen in train | distinct test masked lines seen in train |
| --- | --- | --- |
| Android | 91.5% | 36.6% |
| Apache | 100.0% | 100.0% |
| BGL | 38.1% | 49.0% |
| HDFS | 100.0% | 97.9% |
| HPC | 88.8% | 84.7% |
| Hadoop | 97.5% | 45.9% |
| HealthApp | 99.4% | 57.8% |
| Linux | 97.7% | 92.8% |
| Mac | 93.8% | 65.2% |
| OpenStack | 99.7% | 45.5% |
| Proxifier | 93.4% | 41.7% |
| SSH | 96.5% | 29.1% |
| Spark | 0.7% | 99.5% |
| Thunderbird | 96.9% | 13.3% |
| Zookeeper | 37.1% | 50.4% |
