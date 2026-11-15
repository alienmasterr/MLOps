from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

CFG_PATH = Path(os.environ.get("RAG_CONFIG", "configs/rag.yaml"))


def load(overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(CFG_PATH.read_text())
    for item in overrides or []:
        key, value = item.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for p in path:
            node = node[p]
        node[last] = yaml.safe_load(value)
    if os.environ.get("LLM_BASE_URL"):
        cfg["llm"]["base_url"] = os.environ["LLM_BASE_URL"]
    return cfg


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def corpus_files(cfg: dict) -> list[Path]:
    return sorted(Path(cfg["corpus"]["dir"]).glob(cfg["corpus"]["glob"]))


def corpus_hash(cfg: dict) -> str:
    h = hashlib.sha256()
    for f in corpus_files(cfg):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def index_id(cfg: dict) -> str:
    key = json.dumps([cfg["chunking"], cfg["embedding"]["model"]], sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def config_hash(cfg: dict) -> str:
    safe = json.loads(json.dumps(cfg))
    for s in ("llm", "judge"):
        safe[s].pop("base_url", None)
    return hashlib.sha256(json.dumps(safe, sort_keys=True).encode()).hexdigest()[:12]