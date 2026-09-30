# Embedding sanity report

## Nearest neighbours

Automatic judge on 300 random lines: the nearest neighbour shares the query's Drain3 template in **44.0%** of cases; mean fraction of the 5 nearest neighbours sharing it: 40.4%. (Gate: >= 90% of spot checks. Lines here are *distinct masked strings*, so a neighbour with a different string but the same Drain3 template is the meaningful signal.)

Examples:

- **BGL**: `ciod: In packet from node <NUM> (<LOC>-Ne-C:J11-U11), message code 2 is not <NUM> or <NUM>`
  - NN1: `ciod: In packet from node <NUM> (<LOC>-Ne-C:J05-U11), message code 2 is not <NUM> or <NUM>`
- **BGL**: `Target=<URL> Message=ScomError`
  - NN1: `Target=<URL> Message=Invalid JtagId = ffffffff`
- **BGL**: `r00=<HEX> r01=<HEX> r02=<HEX> r03=<HEX>`
  - NN1: `mask..................<HEX>`
- **BGL**: `ciod: In packet from node <NUM> (<LOC>-Nf-C:J03-U01), message code 2 is not <NUM> or <NUM>`
  - NN1: `ciod: In packet from node <NUM> (<LOC>-Nf-C:J05-U01), message code 2 is not <NUM> or <NUM>`
- **BGL**: `<NUM> torus receiver z+ input pipe error(s) (dcr <HEX>) detected and corrected over <DURAT`
  - NN1: `<NUM> torus receiver z+ input pipe error(s) (dcr <HEX>) detected and corrected`
- **BGL**: `ciod: pollControlDescriptors: Detected the debugger died.`
  - NN1: `<NUM> L3 EDRAM error(s) (dcr <HEX>) detected and corrected over <DURATION:<10ms>`

## t-SNE

![tsne_by_system](figures/tsne_by_system.png)
![tsne_by_level](figures/tsne_by_level.png)

## Linear probe on BGL line labels

Trained on lines seen in the train split, tested on lines whose masked text first appears later (template lookup impossible).

```json
{
  "n_train": 755,
  "n_test_novel": 426,
  "test_pos_rate": 0.07042253521126761,
  "encoder": {
    "pr_auc": 0.4310873546085059,
    "f1": 0.5227272727272727
  },
  "tfidf": {
    "pr_auc": 0.6486285045997676,
    "f1": 0.5862068965517241
  }
}
```
