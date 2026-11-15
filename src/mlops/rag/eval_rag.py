from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import mlflow

from mlops import core
from mlops.rag.config import load
from mlops.rag.llm_client import Meter, chat, parse_score, render
from mlops.rag.retrieve import assert_index_current, retrieve
from mlops.rag.runlog import lineage_tags, log_config


def shingles(txt: str, n: int = 5) -> set[str]:
    w = re.findall(r"\w+", txt.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def title_leaks(cfg: dict, qs: list[dict]) -> list[str]:
    bad = []
    for q in qs:
        doc = Path(cfg["corpus"]["dir"]) / f"{q['relevant_docs'][0]}.md"
        body = doc.read_text()
        title = body.splitlines()[0].lstrip("# ").strip().lower()
        ql = q["question"].lower()
        if title in ql or q["relevant_docs"][0].replace("_", " ") in ql or shingles(ql) & shingles(body):
            bad.append(q["id"])
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--run-name")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    cfg = load(a.override)
    qs = [json.loads(l) for l in Path(cfg["eval"]["questions"]).read_text().splitlines() if l.strip()]
    leaks = title_leaks(cfg, qs)
    print(f"{len(qs)} questions, {len(leaks)} quote their document: {leaks}")
    if a.check_only:
        raise SystemExit(0 if len(qs) >= 20 and not leaks else 1)

    tags = lineage_tags(cfg, a.allow_dirty)
    engine = core.get_engine()
    assert_index_current(engine, cfg)

    k = cfg["retrieval"]["top_k"]
    K = max(5, k)
    ks = sorted({1, 3, 5, k})
    hit = {kk: [] for kk in ks}
    rr_list, scores, bad_cite, no_cite, fails, rows = [], [], [], [], 0, []
    gen, jud = Meter(), Meter()

    for q in qs:
        relevant = set(q["relevant_docs"])
        got = retrieve(engine, cfg, q["question"], K)
        docs = [h["doc_id"] for h in got]
        rank = next((i + 1 for i, d in enumerate(docs) if d in relevant), None)
        rr_list.append(1 / rank if rank else 0.0)
        for kk in ks:
            hit[kk].append(float(rank is not None and rank <= kk))

        ctx = "\n\n".join(f"[{h['doc_id']}] {h['text']}" for h in got[:k])
        r = chat(cfg["llm"], render(cfg["prompts"]["answer"], context=ctx, question=q["question"]))
        gen.add(r)
        cited = set(re.findall(r"\[([A-Za-z0-9_\-]+)\]", r["text"]))
        no_cite.append(float(not cited))
        bad_cite.append(float(bool(cited - set(docs[:k]))))      # посилання на документ, якого retriever не повертав

        j = chat(cfg["judge"], render(cfg["prompts"]["judge"], question=q["question"],
                                      reference=q["reference_answer"], answer=r["text"]))
        jud.add(j)
        sc = parse_score(j["text"], "score")
        if sc is None:
            fails, sc = fails + 1, 0.0
        scores.append(min(sc, 2.0) / 2.0)
        rows.append({"id": q["id"], "question": q["question"], "relevant": sorted(relevant),
                     "retrieved": docs, "rank": rank, "answer": r["text"], "judge": sc})

    mean = lambda v: sum(v) / len(v)
    metr = {f"hit_rate_at_{kk}": mean(hit[kk]) for kk in ks}
    metr.update({"hit_rate_at_k": mean(hit[k]), "mrr": mean(rr_list), "answer_score": mean(scores),
                 "citation_not_retrieved_rate": mean(bad_cite), "no_citation_rate": mean(no_cite),
                 "judge_parse_failures": fails, "n_questions": len(qs), "questions_quoting_doc": len(leaks),
                 **gen.summary(cfg["llm"], "llm"), **jud.summary(cfg["judge"], "judge")})

    name = a.run_name or f"rag-eval-k{k}-c{cfg['chunking']['size_words']}"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("llm-rag")
    Path("reports").mkdir(exist_ok=True)
    detail = Path("reports") / f"{name}.jsonl"
    detail.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({**tags, "kind": "rag_eval"})
        log_config(cfg)
        mlflow.log_metrics(metr)
        mlflow.log_artifact(str(detail))
    print(f"| {name} | k={k} | chunk={cfg['chunking']['size_words']} | hit@k={metr['hit_rate_at_k']:.3f} "
          f"| MRR={metr['mrr']:.3f} | answer={metr['answer_score']:.3f} | run {run.info.run_id} |")


if __name__ == "__main__":
    main()