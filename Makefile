-include .env
export

PY        ?= python
COMPOSE   ?= docker compose
MONTH     ?= 2017-05
SRC       ?= ~/Downloads/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv
MODEL_VERSION ?= 1
AF        = $(COMPOSE) exec -T airflow-scheduler airflow
PSQL      = $(COMPOSE) exec -T postgres psql -U $(POSTGRES_USER) -d warehouse

.PHONY: up down nuke logs sample version-data dvc-creds pull-data \
        trigger trigger-logical backfill break-null break-range trigger-broken count \
        train train-register leaky reproduce

up:
	$(COMPOSE) up -d --build
down:
	$(COMPOSE) down
nuke:
	$(COMPOSE) down -v
logs:
	$(COMPOSE) logs -f --tail=100

sample:
	$(PY) scripts/sample_data.py --src "$(SRC)" --out data/raw --per-month 25000 --seed 42
version-data:
	dvc add data/raw
	dvc push
	@grep md5 data/raw.dvc
dvc-creds:
	dvc remote modify --local minio access_key_id $(MINIO_ROOT_USER)
	dvc remote modify --local minio secret_access_key $(MINIO_ROOT_PASSWORD)
pull-data: dvc-creds
	dvc pull
	@echo "data/raw.dvc checksum:" && grep md5 data/raw.dvc

trigger:
	$(AF) dags trigger toxicity_etl --conf '{"batch_month": "$(MONTH)"}'
trigger-logical:
	$(AF) dags trigger toxicity_etl --logical-date "$(MONTH)-01T00:00:00+00:00"
backfill:
	$(AF) backfill create --dag-id toxicity_etl \
	  --from-date 2016-03-01 --to-date 2017-10-01 --max-active-runs 1
break-null:
	$(PY) scripts/make_broken_batch.py --month $(MONTH) --mode null_text
break-range:
	$(PY) scripts/make_broken_batch.py --month $(MONTH) --mode bad_range
trigger-broken:
	$(AF) dags trigger toxicity_etl \
	  --conf '{"batch_month": "$(MONTH)", "raw_dir": "/opt/airflow/data/broken"}'
count:
	$(PSQL) -c "SELECT 'comments' t, count(*) FROM dwh.comments WHERE batch_month='$(MONTH)' \
	            UNION ALL SELECT 'features', count(*) FROM features.comment_features WHERE batch_month='$(MONTH)'"

train:
	PYTHONPATH=src $(PY) -m mlops.train
train-register:
	PYTHONPATH=src $(PY) -m mlops.train --register
leaky:
	PYTHONPATH=src $(PY) -m mlops.train --leaky-demo
reproduce:
	PYTHONPATH=src $(PY) -m mlops.reproduce --model-version $(MODEL_VERSION) --tol 0.005


LLM_MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
RAGPY     = PYTHONPATH=src $(PY)

.PHONY: llm-up llm-down rag-corpus rag-questions rag-check rag-index \
        llm-zeroshot llm-rag rag-eval rag-sweep-chunk rag-sweep-topk

llm-up:
	docker run -d --rm --name vllm -p 8000:8000 --shm-size=1g \
	  -v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
	  vllm-cpu-env --model $(LLM_MODEL)
llm-down:
	docker stop vllm

rag-corpus:     ; $(RAGPY) scripts/gen_corpus.py
rag-questions:  ; $(RAGPY) scripts/gen_eval_questions.py
rag-check:      ; $(RAGPY) -m mlops.rag.eval_rag --check-only
rag-index:      ; $(RAGPY) -m mlops.rag.ingest

llm-zeroshot:   ; $(RAGPY) -m mlops.rag.classify_llm
llm-rag:        ; $(RAGPY) -m mlops.rag.classify_llm --retrieval
rag-eval:       ; $(RAGPY) -m mlops.rag.eval_rag

rag-sweep-chunk:
	for s in 60 120 240; do \
	  $(RAGPY) -m mlops.rag.ingest --override chunking.size_words=$$s && \
	  $(RAGPY) -m mlops.rag.eval_rag --override chunking.size_words=$$s --run-name chunk-$$s || exit 1; \
	done
	$(RAGPY) -m mlops.rag.ingest        # повернути індекс до значень з configs/rag.yaml
rag-sweep-topk:
	for k in 1 3 5; do \
	  $(RAGPY) -m mlops.rag.eval_rag --override retrieval.top_k=$$k --run-name topk-$$k || exit 1; \
	done