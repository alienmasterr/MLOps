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

## How to run and verify everything (Lab 2 runbook)

Run from the repo root with the virtualenv active. Wait for **Success** in the Airflow UI after every `make trigger`
(a trigger only queues the run, it does not wait for it).

| UI            | URL                   | What it shows                                                 |
| ------------- | --------------------- | ------------------------------------------------------------- |
| Airflow       | http://localhost:8080 | DAG `toxicity_etl`, runs, task graph, task logs               |
| MLflow        | http://localhost:5001 | experiment `toxicity-baseline`, runs, tags, registered models |
| MinIO console | http://localhost:9001 | buckets `dvc` (raw data) and `mlflow` (model artifacts)       |

> MLflow runs on host port **5001** (macOS uses 5000 for AirPlay). PostgreSQL is exposed on host port **5433**;
> inside docker compose services always use `postgres:5432`.

### 0. Start the stack

```bash
cp .env.example .env            # fill in secrets, once
make up                         # postgres, minio (+ buckets), mlflow, airflow
docker compose ps -a            # airflow-init must be "Exited (0)", everything else "Up"
```

- Airflow http://localhost:8080 opens; DAG `toxicity_etl` is listed (unpause it if it is paused).
- MLflow http://localhost:5001 opens (empty).
- MinIO http://localhost:9001 → buckets `dvc` and `mlflow` exist.

### 1. Version the raw data (DVC)

```bash
make sample SRC=/path/to/all_data.csv     # 20 parquet files in data/raw
make version-data                         # dvc add + dvc push, prints md5
grep md5 data/raw.dvc                     # checksum for the README
```

- MinIO http://localhost:9001 → bucket `dvc` now contains the uploaded data.

Clean-clone check:

```bash
git clone <repo-url> /tmp/clean && cd /tmp/clean
cp <original-repo>/.env .env
make pull-data                            # dvc pull
ls data/raw | wc -l                       # 20
grep md5 data/raw.dvc                     # same md5 as above
```

### 2. Run the pipeline for all months (backfill)

```bash
make backfill                             # airflow backfill create, 2016-03 .. 2017-10 (20 runs)
```

- Airflow http://localhost:8080 → DAGs → `toxicity_etl` → Dag Runs: 20 runs turn green one by one.
- Open any green run → Graph: six green tasks (screenshot → `docs/img/dag_success.png`).

Check the warehouse (comments = features for every month):

```bash
docker compose exec postgres psql -U mlops -d warehouse -c "
SELECT c.batch_month, c.n AS comments, f.n AS features
FROM (SELECT batch_month, count(*) n FROM dwh.comments GROUP BY 1) c
LEFT JOIN (SELECT batch_month, count(*) n FROM features.comment_features GROUP BY 1) f USING (batch_month)
ORDER BY 1;"
```

### 3. Idempotency: same month twice, same row count

```bash
make trigger MONTH=2017-05      # wait for Success in Airflow UI
make count MONTH=2017-05        # first COUNT(*)  -> paste into README
make trigger MONTH=2017-05      # wait for Success again
make count MONTH=2017-05        # second COUNT(*) -> must be identical
```

- Airflow UI → Dag Runs: two manual runs for 2017-05, both green.
- Task `load` → Logs: `load: 2017-05 -> N rows` (DELETE + INSERT in one transaction).

### 4. Quality gate fails on purpose

```bash
make break-null MONTH=2017-05   # writes data/broken/2017-05.parquet with 30% null comment_text
make trigger-broken MONTH=2017-05
```

- Airflow UI → the new run is **red**; task `quality_check` is red (screenshot → `docs/img/dag_failed.png`).
- `quality_check` → Logs: line `[FAIL] raw null share of comment_text ...` (paste it into the README).

Restore the good month immediately (otherwise training would use the broken data):

```bash
make trigger MONTH=2017-05      # wait for Success
make count MONTH=2017-05        # back to the original count
```

### 5. Train: rule vs model, tracked in MLflow

The working tree must be clean (training refuses a dirty tree, otherwise the `git_commit` tag would be false):

```bash
git add -A && git status        # no .env, data/broken, data/interim, reports
git commit -m "Lab 2: pipeline, training, docs"
make train                      # prints the rule vs model table
make leaky                      # random split, labelled "leaky", never registered
```

- MLflow http://localhost:5001 → experiment `toxicity-baseline`: runs `rule-badwords`, `tfidf-lr`, `tfidf-lr-RANDOM-SPLIT-LEAKY`.
- Open a run: parameters, metrics (`val_pr_auc`, `val_recall_at_p50`), tags `data_version` and `git_commit`.
- Artifacts tab of `tfidf-lr` is stored in MinIO bucket `mlflow`.

### 6. Register the model with lineage

```bash
make train-register             # registers toxicity-tfidf-lr, tags both identifiers, alias "champion"
git push origin main            # the commit from the tag must exist for reproduce
```

- MLflow http://localhost:5001 → Models → `toxicity-tfidf-lr` → version 1: tags `data_version`, `git_commit`, alias `champion`
  (screenshot of runs and model → `docs/img/mlflow_runs.png`).

### 7. Reproduce the registered model with one command

```bash
make reproduce MODEL_VERSION=1
```

It reads the tags from the registry, checks out that commit in a `git worktree`, runs `dvc pull` (md5 verified), rebuilds the
warehouse months with the same code, retrains and compares `val_pr_auc` with the original (tolerance ±0.005).

- The last line must be `... -> PASS` (paste it into the README).
- MLflow http://localhost:5001 → a new `tfidf-lr` run with tag `reproduces_model_version=1`.

### 8. Final checks before submitting

```bash
grep -n "<<" README.md docs/design.md       # must print nothing (all placeholders filled)
git ls-files | grep -E "^\.env$"            # must print nothing (.env is not committed)
git status                                  # clean
git tag lab-2 && git push origin main --tags
```
