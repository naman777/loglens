"""Typer CLI: `loglens rank <file>`, `loglens watch <paths>`, `loglens bench <file>`."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer

from loglens.serve.runtime import LogLensRuntime, RuntimeConfig, parse_line

app = typer.Typer(add_completion=False, help="LogLens: flag incidents and rank the lines that explain them.")


def _rt(model_dir: str, tokenizer: str, int8: bool, threads: int, cache: bool = True,
        mask_workers: int = 0) -> LogLensRuntime:
    return LogLensRuntime(RuntimeConfig(model_dir=model_dir, tokenizer=tokenizer, int8=int8,
                                        threads=threads, use_cache=cache, mask_workers=mask_workers))


MODEL = typer.Option("artifacts/onnx", help="Directory with encoder/window ONNX files")
TOK = typer.Option("artifacts/tokenizer/loglens-bpe-16k.json", help="Tokenizer json")


def _read(file: Path, frm: str | None, to: str | None) -> list[str]:
    lines = [ln.rstrip("\n") for ln in file.read_text(errors="replace").splitlines() if ln.strip()]
    if not frm and not to:
        return lines
    from loglens.serve.runtime import _ISO, _iso_ms

    def ms(x: str | None):
        m = _ISO.search(x) if x else None
        return _iso_ms(m) if m else None

    lo, hi = ms(frm), ms(to)
    keep = []
    for ln in lines:
        t = parse_line(ln).ts
        if t is None or ((lo is None or t >= lo) and (hi is None or t <= hi)):
            keep.append(ln)
    return keep


@app.command()
def rank(file: Path, top_k: int = 15, frm: str = typer.Option(None, "--from"),
         to: str = typer.Option(None, "--to"), json_out: bool = typer.Option(False, "--json"),
         model_dir: str = MODEL, tokenizer: str = TOK, int8: bool = True, threads: int = 4) -> None:
    """Rank the most suspicious lines in FILE (optionally between --from and --to)."""
    rt = _rt(model_dir, tokenizer, int8, threads)
    res = rt.score(_read(file, frm, to), top_k)
    if json_out or not sys.stdout.isatty():
        typer.echo(json.dumps({"anomaly_score": res.anomaly, "n_lines": res.n_lines,
                               "suspects": res.lines}, indent=2))
        return
    color = typer.colors.RED if res.anomaly > 0.5 else typer.colors.GREEN
    typer.secho(f"{res.n_lines} lines, anomaly score {res.anomaly:.3f}", fg=color, bold=True)
    for s in res.lines:
        typer.echo(f"{s['rank']:>3}  {s['score']:>7.2f}  line {s['index'] + 1:<7} {s['text'][:140]}")


@app.command()
def watch(paths: list[Path], interval: float = 5.0, window: int = 256, top_k: int = 5,
          model_dir: str = MODEL, tokenizer: str = TOK, int8: bool = True, threads: int = 4) -> None:
    """Tail files; every INTERVAL seconds score the latest WINDOW lines and print suspects."""
    rt = _rt(model_dir, tokenizer, int8, threads)
    pos = {p: p.stat().st_size for p in paths}
    buf: list[str] = []
    while True:
        for p in paths:
            with open(p, errors="replace") as f:
                f.seek(pos[p])
                new = f.read()
                pos[p] = f.tell()
            buf += [x for x in new.splitlines() if x.strip()]
        buf = buf[-window:]
        if buf:
            res = rt.score(buf, top_k)
            tag = typer.style("ANOMALY" if res.anomaly > 0.5 else "ok", fg=typer.colors.RED if res.anomaly > 0.5 else typer.colors.GREEN, bold=True)
            typer.echo(f"[{time.strftime('%H:%M:%S')}] {tag} score={res.anomaly:.3f} lines={res.n_lines}")
            if res.anomaly > 0.5:
                for s in res.lines:
                    typer.echo(f"    {s['score']:.2f}  {s['text'][:140]}")
        time.sleep(interval)


@app.command()
def bench(file: Path, max_lines: int = 1_000_000, batch: int = 20_000, model_dir: str = MODEL,
          tokenizer: str = TOK, int8: bool = True, threads: int = 4, cache: bool = True,
          mask_workers: int = 0, cores: int = typer.Option(0, help="pin process tree to N cores (0 = all)")
          ) -> None:
    """Throughput, latency, cache hit rate and peak RAM on FILE."""
    if cores:
        import psutil

        psutil.Process().cpu_affinity(list(range(cores)))
    from loglens.serve import resource_usage

    lines = _read(file, None, None)[:max_lines]
    rt = _rt(model_dir, tokenizer, int8, threads, cache, mask_workers)
    lat = []
    batches = [lines[i: i + batch] for i in range(0, len(lines), batch)]
    t0 = time.perf_counter()
    last = t0
    for _ in rt.score_stream(batches, 15):
        now = time.perf_counter()
        lat.append(now - last)
        last = now
    dt = time.perf_counter() - t0
    import numpy as np

    out = {"lines": len(lines), "lines_per_s": len(lines) / dt, "batch": batch, "threads": threads,
           "int8": int8, "cache": cache, "mask_workers": mask_workers, "cores": cores or "all", "cache_hit_rate": rt.cache.hit_rate,
           "p50_batch_ms": float(np.percentile(lat, 50) * 1000),
           "p99_batch_ms": float(np.percentile(lat, 99) * 1000),
           "peak_rss_mb": resource_usage.peak_rss_mb()}
    typer.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    app()
