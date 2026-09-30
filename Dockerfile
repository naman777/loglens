# LogLens CPU serving image (no PyTorch).
# Build after `make export` so artifacts/onnx and artifacts/tokenizer exist:
#   docker build -t loglens .   &&   docker run --rm -v "$PWD:/data" loglens rank /data/sample.log
FROM python:3.11-slim
WORKDIR /app
COPY loglens/ loglens/
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[serve]" && pip check
COPY artifacts/onnx/ artifacts/onnx/
COPY artifacts/tokenizer/loglens-bpe-16k.json artifacts/tokenizer/loglens-bpe-16k.json
ENV LOGLENS_THREADS=4
ENTRYPOINT ["python", "-m", "loglens.serve.cli"]
