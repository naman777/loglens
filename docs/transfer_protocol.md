# Second unseen-system transfer protocol

## Candidate and provenance

Liberty is a separate supercomputer in the [USENIX HPC4 collection](https://www.usenix.org/cfdr-data),
whose alert/non-alert category tags can support labelled anomaly evaluation. It does not appear in
this repository's tokenizer, encoder or window training corpus. It is from the same collection as
BGL/Thunderbird, so a successful experiment would replicate cross-system transfer, not establish
transfer to an entirely different domain.

The source's HTTPS download failed hostname validation. The legacy HTTP endpoint yielded sample
header lines, but automatic approval review rejected the combined command to create and run a capped
download/preparation script with only "blocked by policy" as the reason. No training-ready sample or
second transfer benchmark was produced. Do not bypass certificate checks or count these probes as
verified dataset acquisition. Obtain a trusted local copy or a verified HTTPS mirror before running.

## Frozen evaluation design

- Record source URI, acquisition date, archive/sample SHA-256, complete versus capped status, parsed
  and rejected line counts, time span and class counts. A hash of a capped prefix is not verification
  of the original full archive. Keep this data under `data/transfer/`, outside corpus globs.
- Strip alert category, epoch/date, host and source fields before constructing model text. The alert
  category supplies only the target label. Empty-message and malformed-line counts must be reported.
- Sort by timestamp, then use chronological 70/10/20 splits without splitting equal timestamps.
  Build nonoverlapping 20-line windows within each split, labelling any-alert windows positive.
  Refuse a headline benchmark if any split lacks either class; document sampling changes first.
- Freeze the existing tokenizer and encoder. Record artifact hashes and verify Liberty did not enter
  their training data. Use the unknown service token rather than borrowing Thunderbird's identity.
- Report zero-shot results at a frozen existing threshold (0.85), and separately a threshold-only
  calibration baseline using the same labelled validation allowance as adaptation.
- Fine-tune the supervised window model using a seeded random 1% of target train windows. Repeat
  for three seeds and report the exact training counts, class balance, selected indices and variance.
- Restrict threshold selection and early stopping to a separately declared validation-label budget.
  The previous Thunderbird experiment used 1% of **training** labels but a larger labelled validation
  set. Do not describe that result as 1% of all labels or compare protocols without disclosing this.
- Compare always-anomalous, a simple logistic-regression baseline at the same train/validation label
  budgets, frozen LogLens, calibrated LogLens and fine-tuned LogLens on identical untouched test windows.
- Report precision, recall, F1, PR-AUC, threshold, sample counts and template overlap. Never tune
  sampling, window size or thresholds using test outcomes.

No numerical Liberty result is currently claimed.
