from __future__ import annotations

import argparse
import json

from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from mlops import core
from mlops.rag.config import config_hash, corpus_files, corpus_hash, index_id, load

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE TABLE IF NOT EXISTS rag.chunks (
    index_id TEXT NOT NULL, chunk_id TEXT NOT NULL, doc_id TEXT NOT NULL,
    text TEXT NOT NULL, embedding vector NOT NULL,
    PRIMARY KEY (index_id, chunk_id));
CREATE TABLE IF NOT EXISTS rag.index_meta (
    index_id TEXT PRIMARY KEY, corpus_hash TEXT NOT NULL, n_docs INT NOT NULL,
    n_chunks INT NOT NULL, config JSONB NOT NULL, built_at TIMESTAMPTZ DEFAULT now());
"""


def chunk_words(txt: str, size: int, overlap: int) -> list[str]:
    w = txt.split()
    step = max(size - overlap, 1)
    return [" ".join(w[i:i + size]) for i in range(0, max(len(w) - overlap, 1), step)]


def vec(e) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in e) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args()
    cfg = load(a.override)

    files = corpus_files(cfg)
    if not files:
        raise SystemExit(f"no documents in {cfg['corpus']['dir']}")
    rows = [(f.stem, f"{f.stem}#{i}", ch) for f in files
            for i, ch in enumerate(chunk_words(f.read_text(), cfg["chunking"]["size_words"],
                                               cfg["chunking"]["overlap_words"]))]
    model = SentenceTransformer(cfg["embedding"]["model"])
    emb = model.encode([r[2] for r in rows], normalize_embeddings=True, show_progress_bar=False)

    iid = index_id(cfg)
    engine = core.get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)
        conn.execute(text("DELETE FROM rag.chunks WHERE index_id = :i"), {"i": iid})
        conn.execute(text("""INSERT INTO rag.chunks (index_id, chunk_id, doc_id, text, embedding)
                             VALUES (:i, :c, :d, :t, CAST(:e AS vector))"""),
                     [{"i": iid, "c": c, "d": d, "t": t, "e": vec(e)} for (d, c, t), e in zip(rows, emb)])
        conn.execute(text("DELETE FROM rag.index_meta WHERE index_id = :i"), {"i": iid})
        conn.execute(text("""INSERT INTO rag.index_meta (index_id, corpus_hash, n_docs, n_chunks, config)
                             VALUES (:i, :h, :nd, :nc, CAST(:cfg AS jsonb))"""),
                     {"i": iid, "h": corpus_hash(cfg), "nd": len(files), "nc": len(rows),
                      "cfg": json.dumps({"chunking": cfg["chunking"], "embedding": cfg["embedding"],
                                         "config_hash": config_hash(cfg)})})
    print(f"indexed {len(files)} documents, {len(rows)} chunks "
          f"(index_id={iid}, corpus_hash={corpus_hash(cfg)})")


if __name__ == "__main__":
    main()