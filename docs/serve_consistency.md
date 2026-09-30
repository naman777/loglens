# Train/serve consistency (BGL raw text through the ONNX runtime)

`scripts/serve_consistency.py` feeds the raw text of BGL test windows (1,200 windows sampled across the
test period, 10.6% positive) through `LogLensRuntime` and compares with the offline result (F1 0.954,
PR-AUC 0.961) at the same validation-tuned threshold.

| runtime parsing | precision | recall | F1 | PR-AUC |
| --- | --- | --- | --- | --- |
| `--format bgl` (header stripped as in training, service `BGL`) | 0.983 | 0.937 | **0.960** | 0.974 |
| generic parser, service guessed (`other`) | 0.891 | 0.709 | 0.789 | 0.768 |
| generic parser, service forced to `BGL` | 0.978 | 0.354 | 0.520 | 0.760 |

**The model was trained on message text only (headers stripped by per-system parsers) with the system as
the service token.** With a matching `--format` the served model reproduces the offline numbers; with
the generic parser, which keeps timestamps/hosts in the message, F1 drops by up to 0.4. For an
arbitrary log format you must either add a header-stripping rule (see `loglens/serve/formats.py`) or
fine-tune on that format's lines. This is a train/serve skew I found late, not a solved problem.
