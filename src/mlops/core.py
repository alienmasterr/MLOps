from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)

FIRST_MONTH = "2016-03"
FEATURE_VERSION = "v1"
DEFAULT_TOXIC_RATE = 0.08

IDENTITY_COLS = ["male", "female", "black", "white", "muslim", "jewish", "christian",
                 "homosexual_gay_or_lesbian", "psychiatric_or_mental_illness"]
REQUIRED_COLUMNS = ["id", "comment_text", "created_date", "publication_id", "parent_id", "rating", "toxicity"]
FEATURE_COLS = ["text_length", "upper_ratio", "exclamation_count", "is_reply",
                "publication_id", "publication_toxic_rate_prev_month"]

QUALITY = {
    "max_raw_null_text_share": 0.05,
    "max_dropped_share": 0.10,
    "min_rows": 5000,
    "toxic_rate_range": (0.02, 0.25),
    "toxicity_range": (0.0, 1.0),
}


class SchemaError(ValueError): ...
class DataQualityError(ValueError): ...


def get_engine():
    return create_engine(os.environ["WAREHOUSE_URL"], future=True)


def interim_path(month: str, name: str) -> Path:
    p = Path(os.environ.get("INTERIM_DIR", "data/interim")) / month
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def previous_month(month: str) -> str:
    return (pd.Period(month, freq="M") - 1).strftime("%Y-%m")


def extract(raw_dir: str | None, month: str) -> pd.DataFrame:
    path = Path(raw_dir or os.environ.get("RAW_DIR", "data/raw")) / f"{month}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"raw batch not found: {path}")
    return pd.read_parquet(path)


def validate_schema(df: pd.DataFrame) -> None:
    t = pd.api.types
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    rules = {
        "id": t.is_integer_dtype,
        "comment_text": lambda s: t.is_string_dtype(s) or t.is_object_dtype(s),
        "created_date": t.is_datetime64_any_dtype,
        "publication_id": t.is_numeric_dtype,
        "parent_id": t.is_numeric_dtype,
        "rating": lambda s: t.is_string_dtype(s) or t.is_object_dtype(s),
        "toxicity": t.is_numeric_dtype,
    }
    bad = [f"{c}: {df[c].dtype}" for c, ok in rules.items() if not ok(df[c])]
    if bad:
        raise SchemaError(f"wrong dtypes: {bad}")


def clean(df: pd.DataFrame, month: str) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"batch_month": month, "n_raw": int(len(df)),
                   "raw_null_text": int(df["comment_text"].isna().sum()), "dropped": {}}
    d = df.copy()

    def drop(mask: pd.Series, rule: str) -> None:
        nonlocal d
        stats["dropped"][rule] = int(mask.sum())
        log.info("cleaning rule %-22s dropped %d rows", rule, stats["dropped"][rule])
        d = d[~mask]

    drop(d["comment_text"].fillna("").astype(str).str.strip().eq(""), "empty_text")
    drop(d["toxicity"].isna(), "null_toxicity")
    drop(d["publication_id"].isna(), "null_publication_id")
    drop(d.duplicated("id"), "duplicate_id")
    drop(d["created_date"].isna() | (d["created_date"].dt.strftime("%Y-%m") != month), "outside_batch_month")

    out = pd.DataFrame({
        "comment_id": d["id"].astype("int64"),
        "batch_month": month,
        "comment_text": d["comment_text"],
        "created_at": d["created_date"],
        "publication_id": d["publication_id"].astype("int64"),
        "parent_id": d["parent_id"].astype("Int64"),
        "rating": d["rating"],
        "toxicity": d["toxicity"].astype(float),
        "is_toxic": d["toxicity"] >= 0.5,     # бінаризація
    })
    for c in IDENTITY_COLS:
        out[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    stats["n_clean"] = int(len(out))
    log.info("transform: %d raw -> %d clean (%s)", stats["n_raw"], stats["n_clean"], stats["dropped"])
    return out.reset_index(drop=True), stats


DDL = """
CREATE SCHEMA IF NOT EXISTS dwh;
CREATE SCHEMA IF NOT EXISTS features;

CREATE TABLE IF NOT EXISTS dwh.comments (
    comment_id     BIGINT NOT NULL,
    batch_month    TEXT NOT NULL,
    comment_text   TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL,
    publication_id INTEGER NOT NULL,
    parent_id      BIGINT,
    rating         TEXT,
    toxicity       DOUBLE PRECISION NOT NULL,   -- навмисно без CHECK: діапазон ловить quality gate
    is_toxic       BOOLEAN NOT NULL,
    male DOUBLE PRECISION, female DOUBLE PRECISION, black DOUBLE PRECISION, white DOUBLE PRECISION,
    muslim DOUBLE PRECISION, jewish DOUBLE PRECISION, christian DOUBLE PRECISION,
    homosexual_gay_or_lesbian DOUBLE PRECISION, psychiatric_or_mental_illness DOUBLE PRECISION,
    PRIMARY KEY (batch_month, comment_id)
);

CREATE TABLE IF NOT EXISTS dwh.publication_month_stats (
    batch_month    TEXT NOT NULL,
    publication_id INTEGER NOT NULL,
    n_comments     INTEGER NOT NULL,
    n_toxic        INTEGER NOT NULL,
    toxic_rate     DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (batch_month, publication_id)
);

CREATE TABLE IF NOT EXISTS features.comment_features (
    comment_id     BIGINT NOT NULL,
    batch_month    TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    text_length    INTEGER NOT NULL,
    upper_ratio    DOUBLE PRECISION NOT NULL,
    exclamation_count INTEGER NOT NULL,
    is_reply       BOOLEAN NOT NULL,
    publication_id INTEGER NOT NULL,
    publication_toxic_rate_prev_month DOUBLE PRECISION NOT NULL,
    rate_source    TEXT NOT NULL,      -- publication | month_global | default
    is_toxic       BOOLEAN NOT NULL,   -- мітка
    PRIMARY KEY (batch_month, comment_id, feature_version)
);
"""


def ensure_schema(engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)


def load(engine, df: pd.DataFrame, month: str) -> int:
    ensure_schema(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM dwh.comments WHERE batch_month = :m"), {"m": month})
        df.to_sql("comments", conn, schema="dwh", if_exists="append",
                  index=False, chunksize=2000, method="multi")
    log.info("load: %s -> %d rows", month, len(df))
    return len(df)


def text_features(s: pd.Series) -> pd.DataFrame:
    t = s.fillna("")
    return pd.DataFrame({
        "text_length": t.str.len(),
        "upper_ratio": t.map(lambda x: sum(ch.isupper() for ch in x) / max(len(x), 1)),
        "exclamation_count": t.str.count("!"),
    })


def build_features(engine, month: str, version: str = FEATURE_VERSION) -> int:
    ensure_schema(engine)
    prev = previous_month(month)
    with engine.begin() as conn:
        c = pd.read_sql(text("""SELECT comment_id, batch_month, comment_text, publication_id, parent_id, is_toxic
                                FROM dwh.comments WHERE batch_month = :m"""), conn, params={"m": month})
        if c.empty:
            raise RuntimeError(f"no comments loaded for {month}")

        st = (c.groupby("publication_id")
                .agg(n_comments=("comment_id", "size"), n_toxic=("is_toxic", "sum")).reset_index())
        st["n_toxic"] = st["n_toxic"].astype(int)
        st["toxic_rate"] = st["n_toxic"] / st["n_comments"]
        st.insert(0, "batch_month", month)
        conn.execute(text("DELETE FROM dwh.publication_month_stats WHERE batch_month = :m"), {"m": month})
        st.to_sql("publication_month_stats", conn, schema="dwh", if_exists="append", index=False)

        ps = pd.read_sql(text("""SELECT publication_id, n_comments, n_toxic, toxic_rate
                                 FROM dwh.publication_month_stats WHERE batch_month = :p"""),
                         conn, params={"p": prev})
        if ps.empty:
            if month != FIRST_MONTH:
                raise RuntimeError(f"previous month {prev} is not loaded: run months in order")
            fallback, fb_src = DEFAULT_TOXIC_RATE, "default"
        else:
            fallback, fb_src = ps["n_toxic"].sum() / ps["n_comments"].sum(), "month_global"

        f = c.merge(ps[["publication_id", "toxic_rate"]], on="publication_id", how="left")
        f["rate_source"] = np.where(f["toxic_rate"].notna(), "publication", fb_src)
        f["publication_toxic_rate_prev_month"] = f["toxic_rate"].fillna(fallback)

        f = pd.concat([f.reset_index(drop=True), text_features(f["comment_text"])], axis=1)
        f["is_reply"] = f["parent_id"].notna()
        f["feature_version"] = version
        cols = ["comment_id", "batch_month", "feature_version", "text_length", "upper_ratio",
                "exclamation_count", "is_reply", "publication_id",
                "publication_toxic_rate_prev_month", "rate_source", "is_toxic"]
        f = f[cols]
        if f.isna().any().any():
            raise RuntimeError("feature table has empty cells: every feature needs a serving-time source")

        conn.execute(text("DELETE FROM features.comment_features WHERE batch_month=:m AND feature_version=:v"),
                     {"m": month, "v": version})
        f.to_sql("comment_features", conn, schema="features", if_exists="append",
                 index=False, chunksize=2000, method="multi")
    log.info("build_features: %s %s -> %d rows", month, version, len(f))
    return len(f)


def quality_check(engine, month: str, stats: dict, version: str = FEATURE_VERSION) -> list[dict]:
    q = QUALITY
    with engine.connect() as conn:
        r = conn.execute(text("""
            SELECT count(*) AS n,
                   count(*) FILTER (WHERE comment_text IS NULL OR btrim(comment_text) = '') AS null_text,
                   min(toxicity) AS tmin, max(toxicity) AS tmax,
                   avg(is_toxic::int) AS toxic_rate
            FROM dwh.comments WHERE batch_month = :m"""), {"m": month}).one()
        n_feat = conn.execute(text("""SELECT count(*) FROM features.comment_features
                                      WHERE batch_month = :m AND feature_version = :v"""),
                              {"m": month, "v": version}).scalar_one()

    raw_null_share = stats["raw_null_text"] / max(stats["n_raw"], 1)
    dropped_share = (stats["n_raw"] - stats["n_clean"]) / max(stats["n_raw"], 1)
    lo, hi = q["toxic_rate_range"]
    checks = [
        ("raw null share of comment_text", raw_null_share <= q["max_raw_null_text_share"],
         f"{raw_null_share:.3f} (max {q['max_raw_null_text_share']})"),
        ("share of rows dropped by cleaning", dropped_share <= q["max_dropped_share"],
         f"{dropped_share:.3f} (max {q['max_dropped_share']})"),
        ("min rows loaded", r.n >= q["min_rows"], f"{r.n} (min {q['min_rows']})"),
        ("nulls in comment_text after load", r.null_text == 0, f"{r.null_text}"),
        ("toxicity within [0, 1]", r.n > 0 and r.tmin >= 0 and r.tmax <= 1, f"min={r.tmin}, max={r.tmax}"),
        ("toxic share within sane bounds", r.n > 0 and lo <= r.toxic_rate <= hi, f"{r.toxic_rate:.3f} in [{lo}, {hi}]"),
        ("feature rows == comment rows", n_feat == r.n, f"features={n_feat}, comments={r.n}"),
    ]
    result = [{"check": n, "passed": bool(ok), "detail": d} for n, ok, d in checks]
    for c in result:
        log.info("[%s] %s: %s", "PASS" if c["passed"] else "FAIL", c["check"], c["detail"])
    failed = [c for c in result if not c["passed"]]
    if failed:
        raise DataQualityError(f"quality gate failed for {month}: " +
                               "; ".join(f"{c['check']} -> {c['detail']}" for c in failed))
    return result