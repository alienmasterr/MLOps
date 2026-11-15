from __future__ import annotations

import mlflow

from mlops.rag.config import CFG_PATH, config_hash, corpus_hash, file_hash
from mlops.train import assert_clean_tree, data_version, git


def lineage_tags(cfg: dict, allow_dirty: bool = False) -> dict:
    if not allow_dirty:
        assert_clean_tree()
    tags = {"component": "llm", "git_commit": git("rev-parse", "HEAD"),
            "data_version": data_version(), "corpus_hash": corpus_hash(cfg),
            "config_hash": config_hash(cfg)}
    for name, path in cfg["prompts"].items():
        tags[f"prompt_hash_{name}"] = file_hash(path)
    return tags


def log_config(cfg: dict) -> None:
    mlflow.log_params({
        "llm_model": cfg["llm"]["model"], "llm_hosting": cfg["llm"]["hosting"],
        "temperature": cfg["llm"]["temperature"], "max_tokens": cfg["llm"]["max_tokens"],
        "top_k": cfg["retrieval"]["top_k"], "chunk_size_words": cfg["chunking"]["size_words"],
        "chunk_overlap_words": cfg["chunking"]["overlap_words"],
        "embedding_model": cfg["embedding"]["model"], "judge_model": cfg["judge"]["model"],
        "seed": cfg["eval"]["seed"]})
    mlflow.log_artifact(str(CFG_PATH))
    for p in cfg["prompts"].values():
        mlflow.log_artifact(p, "prompts")