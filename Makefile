PY ?= python

.PHONY: setup data lab-up lab-down lab-campaign tokenizer pretrain finetune eval export serve test lint

setup:
	$(PY) -m pip install -e ".[dev,train,serve,baselines]"

data:
	$(PY) -m loglens.data.build --config configs/data.yaml

lab-up:
	docker compose -f lab/docker-compose.yml up -d --build

lab-down:
	docker compose -f lab/docker-compose.yml down -v

lab-campaign:
	$(PY) lab/run_campaign.py && $(PY) -m loglens.data.lab

tokenizer:
	$(PY) -m loglens.tokenizer.train_bpe --config configs/tokenizer.yaml

pretrain:
	$(PY) -m loglens.train.pretrain --config configs/pretrain.yaml

finetune:
	$(PY) -m loglens.train.finetune --config configs/finetune.yaml

eval:
	$(PY) -m loglens.eval.report --config configs/eval.yaml

export:
	$(PY) -m loglens.serve.export_onnx --config configs/serve.yaml

serve:
	$(PY) -m uvicorn loglens.serve.api:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
