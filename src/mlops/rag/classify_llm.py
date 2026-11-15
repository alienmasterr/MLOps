from __future__ import annotations

import argparse
import os

import mlflow
from mlflow import sklearn as mlflow_sklearn

from mlops import core
from mlops.rag.config import load
from mlops.rag.llm_client import Meter, chat, parse_score, render
from mlops.rag.retrieve import assert_index_current, retrieve
from mlops.rag.runlog import lineage_tags, log_config
from mlops.rules import rule_score
from mlops.train import CAT, MODEL_NAME, NUM, VAL, load_frame, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    cfg = load(a.override)
    tags = lineage_tags(cfg, a.allow_dirty)
    engine = core.get_engine()
    if a.retrieval:
        assert_index_current(engine, cfg)

    val = load_frame(engine, *VAL)
    s = val.sample(n=min(cfg["eval"]["classify_n"], len(val)),
                   random_state=cfg["eval"]["seed"]).reset_index(drop=True)
    y = s["is_toxic"].astype(int)

    template = cfg["prompts"]["classify_rag" if a.retrieval else "classify"]
    meter, scores, fails = Meter(), [], 0
    for i, comment in enumerate(s["comment_text"]):
        ctx = ""
        if a.retrieval:
            ctx = "\n\n".join(f"[{h['doc_id']}] {h['text']}"
                              for h in retrieve(engine, cfg, comment[:500]))
        r = chat(cfg["llm"], render(template, comment=comment[:1500], context=ctx))
        meter.add(r)
        sc = parse_score(r["text"], "toxicity_score")
        if sc is None:
            fails, sc = fails + 1, 0.0
        scores.append(min(max(sc, 0.0), 1.0))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(s)} done", flush=True)

    m_llm = metrics(y, scores)
    m_rule = metrics(y, s["comment_text"].map(rule_score).to_numpy())
    champ = mlflow_sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    m_champ = metrics(y, champ.predict_proba(s[NUM + CAT + ["comment_text"]])[:, 1])

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("llm-rag")
    name = f"llm-{'rag' if a.retrieval else 'zeroshot'}-{cfg['llm']['model'].split('/')[-1]}"
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({**tags, "kind": "llm_classifier", "retrieval": str(a.retrieval)})
        log_config(cfg)
        mlflow.log_params({"n_eval": len(s), "split": "validation 2017-04..05 (sample)"})
        mlflow.log_metrics({
            "val_pr_auc": m_llm["pr_auc"],
            "val_recall_at_p50": m_llm["recall_at_p50"],
            "same_sample_rule_pr_auc": m_rule["pr_auc"],
            "same_sample_rule_recall_at_p50": m_rule["recall_at_p50"],
            "same_sample_champion_pr_auc": m_champ["pr_auc"],
            "same_sample_champion_recall_at_p50": m_champ["recall_at_p50"],
            "parse_failures": fails,
            **meter.summary(cfg["llm"], "llm"),
        })

    print("\n| (same sample) | PR-AUC | recall@precision>=0.50 |\n|---|---|---|")
    print(f"| rule | {m_rule['pr_auc']:.4f} | {m_rule['recall_at_p50']:.4f} |")
    print(f"| TF-IDF+LogReg (champion) | {m_champ['pr_auc']:.4f} | {m_champ['recall_at_p50']:.4f} |")
    print(f"| LLM {name} | {m_llm['pr_auc']:.4f} | {m_llm['recall_at_p50']:.4f} |")
    print(f"parse failures: {fails}/{len(s)}")
    print("run id:", run.info.run_id)


if __name__ == "__main__":
    main()