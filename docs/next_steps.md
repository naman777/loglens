# Next validation work

## Implemented in the current follow-up

- Real Postgres/Redis interactions, payment connection holds, worker retry/acknowledgement and HTTP
  failure propagation; bounded spool storage and service health checks.
- Six reversible real fault types and an evidence-producing normal/fault/recovery runner.
- A Docker CI job that uploads campaign evidence. It is configured, not yet executed here.
- Paired single-call LLM diagnosis evaluation on held-out synthetic incidents, with equal prompt
  budgets, invalid-answer accounting, token counts, latency and persisted answers.

See `real_lab.md` for execution instructions. Docker is still absent locally, so local mocked/service
unit tests must not be represented as successful real-container validation.

## Remaining experiment and release work

1. Run the Docker campaign in CI or on a Docker-enabled machine; inspect failed injections, review
   causal evidence across services, and keep real results separate from simulator benchmarks.
2. Expand diagnosis evaluation to real incidents and stronger models, then a genuine tool-using agent.
   The local small-model experiment measures budget-constrained single-call answers only.
3. Repeat few-label transfer on another independently held-out labelled system. All other datasets
   currently processed were exposed during encoder/tokenizer training, so simply reusing them would
   not establish unseen-system transfer. Either obtain a new labelled system or retrain with an
   additional exclusion from every training stage; record provenance and label budgets explicitly.
4. Publish matching weights and test the full training/inference quickstart from a fresh clone.
5. Improve threshold stability and unfamiliar-format parsing, followed by throughput optimization.

No Docker benchmark, additional unseen-system score or public release is implied by these changes.
