from __future__ import annotations

from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from mlops.rag.config import corpus_hash, index_id

_models: dict[str, SentenceTransformer] = {}


def _embedder(name: str) -> SentenceTransformer:
    if name not in _models:
        _models[name] = SentenceTransformer(name)
    return _models[name]


def assert_index_current(engine, cfg: dict) -> None:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT corpus_hash FROM rag.index_meta WHERE index_id = :i"),
                           {"i": index_id(cfg)}).first()
    if row is None:
        raise SystemExit("index not built for this config: run `make rag-index`")
    if row[0] != corpus_hash(cfg):
        raise SystemExit(f"index is stale (corpus changed: {row[0]} -> {corpus_hash(cfg)}): run `make rag-index`")


def retrieve(engine, cfg: dict, query: str, k: int | None = None) -> list[dict]:
    k = k or cfg["retrieval"]["top_k"]
    e = _embedder(cfg["embedding"]["model"]).encode([query], normalize_embeddings=True)[0]
    q = "[" + ",".join(f"{x:.6f}" for x in e) + "]"
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT chunk_id, doc_id, text, 1 - (embedding <=> CAST(:e AS vector)) AS score
            FROM rag.chunks WHERE index_id = :i
            ORDER BY embedding <=> CAST(:e AS vector) LIMIT :k"""),
            {"e": q, "i": index_id(cfg), "k": k}).mappings().all()
    return [dict(r) for r in rows]