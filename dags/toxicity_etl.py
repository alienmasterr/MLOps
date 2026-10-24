from __future__ import annotations

import json
import logging

import pandas as pd
import pendulum
from airflow.sdk import dag, get_current_context, task
from airflow.timetables.interval import CronDataIntervalTimetable

from mlops import core

log = logging.getLogger(__name__)


def _batch_month() -> str:
    ctx = get_current_context()
    override = (ctx["params"].get("batch_month") or "").strip()
    if override:
        return override
    month = ctx["data_interval_start"].strftime("%Y-%m")
    log.info("data_interval = [%s, %s) -> batch_month=%s",
             ctx["data_interval_start"], ctx["data_interval_end"], month)
    return month


@dag(
    dag_id="toxicity_etl",
    schedule=CronDataIntervalTimetable("@monthly", timezone="UTC"),
    start_date=pendulum.datetime(2016, 3, 1, tz="UTC"),
    end_date=pendulum.datetime(2017, 10, 31, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    params={"batch_month": "", "raw_dir": ""},
    tags=["lab2"],
)
def toxicity_etl():
    @task
    def extract() -> dict:
        month = _batch_month()
        raw_dir = (get_current_context()["params"].get("raw_dir") or "").strip() or None
        df = core.extract(raw_dir, month)
        df.to_parquet(core.interim_path(month, "extracted.parquet"), index=False)
        log.info("extract: %s -> %d rows", month, len(df))
        return {"batch_month": month, "n_extracted": len(df)}

    @task
    def validate(ctx: dict) -> dict:
        df = pd.read_parquet(core.interim_path(ctx["batch_month"], "extracted.parquet"))
        core.validate_schema(df)
        log.info("validate: schema OK (%d columns)", df.shape[1])
        return ctx

    @task
    def transform(ctx: dict) -> dict:
        month = ctx["batch_month"]
        df = pd.read_parquet(core.interim_path(month, "extracted.parquet"))
        clean, stats = core.clean(df, month)
        clean.to_parquet(core.interim_path(month, "clean.parquet"), index=False)
        core.interim_path(month, "stats.json").write_text(json.dumps(stats))
        return {**ctx, "stats": stats}

    @task
    def load(ctx: dict) -> dict:
        month = ctx["batch_month"]
        df = pd.read_parquet(core.interim_path(month, "clean.parquet"))
        n = core.load(core.get_engine(), df, month)
        return {**ctx, "n_loaded": n}

    @task
    def build_features(ctx: dict) -> dict:
        n = core.build_features(core.get_engine(), ctx["batch_month"])
        return {**ctx, "n_features": n}

    @task
    def quality_check(ctx: dict) -> None:
        core.quality_check(core.get_engine(), ctx["batch_month"], ctx["stats"])

    quality_check(build_features(load(transform(validate(extract())))))


toxicity_etl()