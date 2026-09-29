# LogLens

A small log-intelligence model, trained from scratch, that flags incidents and ranks the log lines
that explain them — on a laptop CPU, with zero LLM API calls.

Status: under active construction. See [progress.md](progress.md) for the live build log and
`docs/` for reports. Benchmark table will land here once Phase 6 finishes.

## Quickstart

```bash
pip install -e ".[dev,train,serve,baselines]"
make test
```
