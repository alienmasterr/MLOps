from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import text

from mlops import core
from mlops.rules import rule_score

EXPERIMENT = "toxicity-baseline"
MODEL_NAME = "toxicity-tfidf-lr"
FEATURE_VERSION = core.FEATURE_VERSION
TRAIN = ("2016-03", "2017-03")     # 13 місяців
VAL = ("2017-04", "2017-05")       # 2 місяці; test 2017-09..10 НЕ чіпаємо
NUM = ["text_length", "upper_ratio", "exclamation_count", "is_reply", "publication_toxic_rate_prev_month"]
CAT = ["publication_id"]
P_TARGET = 0.50


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def assert_clean_tree() -> None:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        sys.exit("Git tree is dirty: git_commit tag would be false. Commit first (or --allow-dirty).\n" + dirty)
    if subprocess.run(["dvc", "status", "--quiet"]).returncode != 0:
        sys.exit("data/raw differs from data/raw.dvc: run `dvc checkout` / `dvc add`.")


def data_version() -> str:
    return yaml.safe_load(Path("data/raw.dvc").read_text())["outs"][0]["md5"]


def load_frame(engine, first: str, last: str) -> pd.DataFrame:
    q = text("""SELECT f.*, c.comment_text
                FROM features.comment_features f
                JOIN dwh.comments c USING (batch_month, comment_id)
                WHERE f.feature_version = :v AND f.batch_month BETWEEN :a AND :b
                ORDER BY f.batch_month, f.comment_id""")
    df = pd.read_sql(q, engine, params={"v": FEATURE_VERSION, "a": first, "b": last})
    df[NUM] = df[NUM].astype(float)
    return df


def recall_at_precision(y, s, p: float = P_TARGET) -> float:
    prec, rec, _ = precision_recall_curve(y, s)
    m = prec >= p
    return float(rec[m].max()) if m.any() else 0.0


def build_model(C: float, max_features: int, seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features, sublinear_tf=True),
         "comment_text"),
        ("num", StandardScaler(), NUM),
        ("pub", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def metrics(y, s) -> dict:
    return {"pr_auc": float(average_precision_score(y, s)), "recall_at_p50": recall_at_precision(y, s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-features", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--leaky-demo", action="store_true", help="random split: ЛИШЕ для порівняння, не реєструється")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--extra-tag", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()

    if not a.allow_dirty:
        assert_clean_tree()
    lineage = {"data_version": data_version(), "git_commit": git("rev-parse", "HEAD")}
    extra = dict(t.split("=", 1) for t in a.extra_tag)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT)
    engine = core.get_engine()
    client = MlflowClient()

    if a.leaky_demo:
        df = load_frame(engine, TRAIN[0], VAL[1])
        tr, va = train_test_split(df, test_size=0.2, random_state=a.seed, stratify=df["is_toxic"])
        model = build_model(a.C, a.max_features, a.seed).fit(tr[NUM + CAT + ["comment_text"]], tr["is_toxic"])
        m = metrics(va["is_toxic"], model.predict_proba(va[NUM + CAT + ["comment_text"]])[:, 1])
        with mlflow.start_run(run_name="tfidf-lr-RANDOM-SPLIT-LEAKY"):
            mlflow.set_tags({**lineage, "split": "random_LEAKY", "registered": "never"})
            mlflow.log_params({"seed": a.seed, "feature_version": FEATURE_VERSION, "split": "random 80/20"})
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
        print("LEAKY (random split):", m)
        return

    train, val = load_frame(engine, *TRAIN), load_frame(engine, *VAL)
    X_cols = NUM + CAT + ["comment_text"]
    y_tr, y_va = train["is_toxic"], val["is_toxic"]
    common_params = {"seed": a.seed, "feature_version": FEATURE_VERSION,
                     "train_months": "%s..%s" % TRAIN, "val_months": "%s..%s" % VAL,
                     "n_train": len(train), "n_val": len(val)}

    s_rule = val["comment_text"].map(rule_score).to_numpy()
    m_rule = metrics(y_va, s_rule)
    flagged = s_rule > 0
    with mlflow.start_run(run_name="rule-badwords") as r_rule:
        mlflow.set_tags({**lineage, "kind": "rule", **extra})
        mlflow.log_params({**common_params, "n_bad_words": len(__import__("mlops.rules").rules.BAD_WORDS)})
        mlflow.log_metrics({f"val_{k}": v for k, v in m_rule.items()} |
                           {"val_precision_at_rule": float(precision_score(y_va, flagged)),
                            "val_recall_at_rule": float(recall_score(y_va, flagged))})

    model = build_model(a.C, a.max_features, a.seed).fit(train[X_cols], y_tr)
    proba = model.predict_proba(val[X_cols])[:, 1]
    m_model = metrics(y_va, proba)
    with mlflow.start_run(run_name="tfidf-lr") as run:
        mlflow.set_tags({**lineage, "kind": "model", **extra})
        mlflow.log_params({**common_params, "C": a.C, "max_features": a.max_features,
                           "class_weight": "balanced", "ngram_range": "1-2", "numeric_features": ",".join(NUM)})
        mlflow.log_metrics({f"val_{k}": v for k, v in m_model.items()})
        info = mlflow.sklearn.log_model(
            model, name="model",
            signature=infer_signature(val[X_cols].head(100), proba[:100]),
            registered_model_name=MODEL_NAME if a.register else None,
        )
        if a.register:
            v = str(info.registered_model_version)
            for k, val_ in lineage.items():
                client.set_model_version_tag(MODEL_NAME, v, k, val_)
            client.set_registered_model_alias(MODEL_NAME, a.alias, v)
            print(f"registered {MODEL_NAME} v{v}, alias '{a.alias}', tags {lineage}")

    print("\n| | PR-AUC | recall@precision>=0.50 |\n|---|---|---|")
    print(f"| rule (bad words) | {m_rule['pr_auc']:.4f} | {m_rule['recall_at_p50']:.4f} |")
    print(f"| TF-IDF + LogReg  | {m_model['pr_auc']:.4f} | {m_model['recall_at_p50']:.4f} |")
    print(f"run ids: rule={r_rule.info.run_id} model={run.info.run_id}")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/last_run.json").write_text(json.dumps(
        {"run_id": run.info.run_id, "rule_run_id": r_rule.info.run_id, **m_model}))


if __name__ == "__main__":
    main()