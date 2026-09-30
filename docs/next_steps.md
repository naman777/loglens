# Next validation work

## Implemented in the current follow-up

- Real Postgres/Redis interactions, payment connection holds, worker retry/acknowledgement and HTTP
  failure propagation; bounded spool storage and service health checks.
- Six reversible real fault types and an evidence-producing normal/fault/recovery runner.
- A Docker CI job with six fault/recovery checks passing and preserved evidence (run 36725870364).
- Paired single-call LLM diagnosis evaluation on held-out synthetic incidents, with equal prompt
  budgets, invalid-answer accounting, token counts, latency and persisted answers.

See `real_lab.md` and `real_lab_validation.md` for commands and actual CI evidence. Docker remains
absent locally; the successful real-container validation was performed on the Linux CI runner.

## Remaining experiment and release work

1. Expand beyond the six-case Docker smoke campaign and obtain human-reviewed causal labels.
   Keep real results separate from simulator benchmarks.
2. Expand the six-case real diagnosis pilot to more incidents and stronger models, then a genuine
   tool-using agent. Current single-call results: synthetic 31/36 vs 6/36; real 3/6 vs 1/6.
3. Repeat few-label transfer on another independently held-out labelled system. All other datasets
   currently processed were exposed during encoder/tokenizer training, so simply reusing them would
   not establish unseen-system transfer. Either obtain a new labelled system or retrain with an
   additional exclusion from every training stage; record provenance and label budgets explicitly.
4. Finish the matching-weight prerelease and fresh-clone inference verification. Full fresh-clone
   training reproduction remains separate and unverified.
5. Improve threshold stability and unfamiliar-format parsing. Format-specific four-core throughput
   now exceeds 20k lines/s; generic-parser optimization remains open.

See `transfer_protocol.md` for the second-system design and the blocked Liberty preparation attempt.
No additional unseen-system accuracy score or production-readiness claim is made.
