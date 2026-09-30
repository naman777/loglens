FROM python:3.11-slim
ARG SERVICE
ENV SERVICE=$SERVICE
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx "psycopg[binary]" redis
COPY common.py database.py ${SERVICE}.py ./
CMD ["sh", "-c", "uvicorn ${SERVICE}:app --host 0.0.0.0 --port ${PORT:-8000} --log-level warning"]
