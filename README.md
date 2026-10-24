# MLOps

| Property                                     | Evidence a reviewer checks                                                               | Added in |
| :------------------------------------------- | :--------------------------------------------------------------------------------------- | :------- |
| _Repository reproducible from a clean clone_ | _CI is green;_ `make setup` _and_ `make lint` _succeed on a fresh clone_                 | _Lab 1_  |
| _Design document exists and is current_      | `docs/design.md` _follows the template, every section filled or marked as an assumption_ | _Lab 1_  |

Online platforms receive far more comments than human moderators can read, and the language of toxic comments keeps changing.
This project builds a system that scores every new comment for toxicity and decides: publish, send to human review, or hide.
It uses the Jigsaw Unintended Bias (Civil Comments) dataset, split by time so that new months can be replayed as new batches.
The system watches its own inputs and predictions for drift, and a human approves any retrained model before it goes live.
The goal is to catch most toxic comments quickly while keeping the human review queue small and the whole system reproducible from this repository.

Full design: docs/design.md

## Lab 2: data pipelines, versioning, baseline, MLflow

### Quick start (from a clean clone)

```bash
cp .env.example .env        # fill in secrets; keep MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false
make up                     # postgres, minio (+ buckets dvc, mlflow), mlflow, airflow
make pull-data              # DVC pull from MinIO -> data/raw
```

UIs: Airflow http://localhost:8080 · MLflow http://localhost:5001 · MinIO console http://localhost:9001

### Data versioning (DVC)

- Remote type: MinIO (S3-compatible), bucket dvc, image pgsty/silo:RELEASE.2026-09-16T00-00-00Z.
  Remote config is committed (.dvc/config), keys live in .dvc/config.local (gitignored; make dvc-creds writes them from .env).
- Retrieve on a clean clone: make pull-data (= dvc pull).

### Pipeline (Airflow DAG toxicity_etl)

extract -> validate -> transform -> load -> build_features -> quality_check

Interval choice: schedule=CronDataIntervalTimetable("@monthly", timezone="UTC"), so
data_interval_start =! data_interval_end; batch = data_interval_start formatted YYYY-MM.
Optional override: --conf '{"batch_month": "YYYY-MM"}'. The batch is never taken from the wall clock.

Run a month / backfill:

```bash
make trigger MONTH=2017-05
make trigger-logical MONTH=2017-05
make backfill
```

Idempotency (load runs in one transaction: DELETE FROM dwh.comments WHERE batch_month = :m, then INSERT):

```text
$ make trigger MONTH=2017-05 && make count MONTH=2017-05
 comments | <<N>>        features | <<N>>
$ make trigger MONTH=2017-05 && make count MONTH=2017-05
 comments | <<N>>        features | <<N>>        <- identical
```

### Baseline on a time-based split

Train 2016-03 -> 2017-03, validation 2017-04 -> 2017-05, test 2017-09 -> 2017-10 untouched.

|                                         | PR-AUC | recall @ precision ≥ 0.50 |
| --------------------------------------- | ------ | ------------------------- |
| Rule (50 bad words)                     | <<…>>  | <<…>>                     |
| TF-IDF + LogReg (class_weight=balanced) | <<…>>  | <<…>>                     |
| (leaky, random split, not registered)   | <<…>>  | <<…>>                     |

which one won and why:

### Reproduce a registered model

```bash
make reproduce MODEL_VERSION=1
```
