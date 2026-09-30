# Error analysis

Generated from `scripts/error_analysis.py` -> `results/error_analysis.json` (worst misses of the
supervised LogLens model). Small samples; these are qualitative findings, not statistics.

## Lab root-cause ranking (36 test incidents, 2 misses at Recall@5)

Both misses are `container_kill` incidents (the target's first causal line ranks 551 and 874 of
~2,300 lines):

- The killed service emits **no logs while it is down**; the only lines the rule labeller marks causal
  are the two restart INFO lines (`<svc> service starting, version <VER>`, `listening on port <NUM>,
  ready to serve`). Nothing about them looks anomalous in isolation, so the suspicion head prefers noisy
  WARN lines (`slow query ... took <DURATION:<1s>`, `deprecated api version header`).
- The *dependents'* `connection refused calling <svc>` lines are the strongest real evidence, but the
  labelling rule (target-service lines only) does not count them, so the model is never rewarded for
  ranking them first. This is a labelling-design limit, not only a model limit: a service-level
  attribution metric (see `results/agent_eval.json`) would credit those lines.
- The other 34 incidents (including `bad_config`, which also restarts the service) reach Recall@5.

## BGL anomaly detection (test: 66 false negatives, 5 false positives at the tuned threshold)

- **False negatives are rare alert templates**: `No power module U58 found found on link card`,
  `ciodb exited abnormally due to signal: Aborted`, Java `IllegalStateException`/`EOFException` monitor
  errors. They have few or no train-time occurrences (BGL is a non-stationary log: only 38% of test lines
  have a masked text seen in train, see `data_report.md`), and a 256-line window dilutes a single rare
  line among ~255 benign ones.
- **False positives are high-frequency benign-looking templates that co-occur with alerts in train**:
  `<NUM> ddr errors(s) detected and corrected`, `L1 DCACHE summary averages`, `<NUM> microseconds spent
  in the rbs signal handler`. The label is "any alert line in the window", so these templates absorbed
  signal from neighbouring alerts during training.

## Unseen system (Thunderbird)

- Zero-shot LogLens (supervised on BGL/HDFS/lab, or unsupervised) does **not** transfer: F1 equals the
  always-anomalous baseline and PR-AUC is at or below it. The alert taxonomy differs per system and
  the pretrained line encoder does not bridge it (the random-init-encoder ablation transfers just as
  poorly, see `results/ablations.md`).
- With 1% of Thunderbird's labels (2,100 windows) fine-tuning reaches F1 0.985, above a logistic
  regression trained on 100% of its labels (F1 0.858). This is the useful transfer result; treat it
  cautiously (one system, one split).

## Things that did not work / were not validated

- **SimCSE contrastive loss hurt embedding quality** (nearest neighbour shares the Drain3 template 44%
  vs 68% without): with distinct masked lines as instances, in-batch negatives push apart lines of the
  same template.
- **Line-encoder pretraining is not measurably better than a random-init encoder in-distribution**
  (`results/ablations.md`): with ~1k templates per system a random projection is already injective.
  The plan's linear-probe gate (encoder beats TF-IDF) and the 90% nearest-neighbour gate were not met.
- **Tokenizer**: 1.5x average (1.3x on Thunderbird) fewer tokens than raw GPT-2, below the 2x gate.
- **Masking** originally left 65,536 distinct `core.<N>` lines in BGL; found via digit-collapse
  statistics, fixed, and everything downstream re-run.

## Template-contrastive encoder (v3) and masking ablation

- Adding a supervised-contrastive loss over Drain3 templates raised nearest-neighbour agreement only
  from 68% to 72% (gate 90%) and the linear probe still loses to TF-IDF. Downstream it gave the best
  zero-shot Thunderbird PR-AUC so far (0.70 vs 0.56) and RCA Recall@5 1.00, but BGL F1 collapsed to
  0.47 while BGL PR-AUC stayed 0.96, i.e. the validation-tuned threshold did not transfer across the
  BGL time shift. Single run, single seed: not adopted as the main model.
- Collapsing durations to `<NUM>` (random-init encoder) lowered BGL F1 0.939 -> 0.872 and HDFS 0.995 ->
  0.950: keeping bucketed durations helps.
