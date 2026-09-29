# LogLens CPU serving image: onnxruntime + tokenizers only (no PyTorch).
# Build after `make export` so artifacts/onnx and artifacts/tokenizer exist:
#   docker build -t loglens .   &&   docker run --rm -v "$PWD:/data" loglens rank /data/sample.log
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir onnxruntime tokenizers numpy typer fastapi "uvicorn[standard]" psutil pyyaml
COPY loglens/ loglens/
COPY pyproject.toml README.md ./
COPY artifacts/onnx/ artifacts/onnx/
COPY artifacts/tokenizer/loglens-bpe-16k.json artifacts/tokenizer/loglens-bpe-16k.json
RUN pip install --no-cache-dir --no-deps -e .
ENV LOGLENS_THREADS=4
ENTRYPOINT ["python", "-m", "loglens.serve.cli"]
