PY ?= python

.PHONY: setup data lab-up lab-down lab-campaign masked tokenizer pretrain embed finetune eval export bench serve test lint

setup:
	$(PY) -m pip install -e ".[dev,train,serve,export,baselines,lab]"

# download Loghub (Zenodo) -> parse -> time-split Parquet; Thunderbird is streamed (first 6M lines)
data:
	$(PY) -m loglens.data.download
	$(PY) -m loglens.data.download_partial Thunderbird 6000000
	$(PY) -m loglens.data.build --config configs/data.yaml

lab-up:
	docker compose -f lab/docker-compose.yml up -d --build

lab-down:
	docker compose -f lab/docker-compose.yml down -v

# simulated campaigns (no Docker needed) + rule-labelled causal lines -> data/lab, data/parquet/Lab
lab-campaign:
	$(PY) lab/run_campaign.py && $(PY) -m loglens.data.lab

# mask every line once, then build windows
masked:
	$(PY) -m loglens.data.masked
	$(PY) -m loglens.data.windows

tokenizer:
	$(PY) -m loglens.tokenizer.train_bpe --config configs/tokenizer.yaml
	$(PY) -m loglens.tokenizer.evaluate
	$(PY) -m loglens.data.pretok

pretrain:
	$(PY) -m loglens.train.pretrain --config configs/pretrain.yaml

embed:
	$(PY) -m loglens.train.embed

finetune:
	$(PY) -m loglens.train.finetune --config configs/finetune_sup.yaml
	$(PY) -m loglens.train.finetune --config configs/finetune_unsup.yaml

eval:
	$(PY) -m loglens.eval.report --config configs/eval.yaml

export:
	$(PY) -m loglens.serve.export_onnx --config configs/serve.yaml
	$(PY) -m loglens.serve.int8_eval

bench:
	$(PY) -m loglens.serve.cli bench data/bench/bgl_1m.log --mask-workers 3 --cores 4

serve:
	$(PY) -m uvicorn loglens.serve.api:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
