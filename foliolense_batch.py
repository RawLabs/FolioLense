#!/usr/bin/env python3
"""
foliolense_batch.py

Batch-runs the health-pattern query pack against the existing FolioLense SQLite index.

Important optimization:
- loads the page matrix once
- loads Qwen3-VL-Embedding-2B-vdr once
- encodes all queries in one model session
- writes JSONL compatible with health_pattern_runner.py queue

Usage:
  ~/.local/share/foliolense/.venv/bin/python foliolense_batch.py \
    --queries health_queries.jsonl \
    --db ~/.local/share/foliolense/index.sqlite \
    --output foliolense_results.jsonl \
    --top 15
"""

from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


MODEL_ID = "tomaarsen/Qwen3-VL-Embedding-2B-vdr"


def connect_db(path: Path):
    conn = sqlite3.connect(path.expanduser().resolve())
    return conn


def get_meta(conn, key):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def load_matrix(conn):
    has_file_paths = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='file_paths'"
    ).fetchone()
    if has_file_paths:
        rows = conn.execute(
            """
            SELECT p.id,
                   COALESCE(
                       (SELECT fp.path FROM file_paths fp
                        WHERE fp.sha256 = p.file_sha256
                        ORDER BY fp.last_seen DESC, fp.path ASC LIMIT 1),
                       '[sha256:' || substr(p.file_sha256, 1, 12) || ']'
                   ),
                   p.page, p.file_sha256, p.dim, p.embedding
            FROM pages p ORDER BY p.id
            """
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id,path,page,file_sha256,dim,embedding FROM pages ORDER BY id"
        ).fetchall()
    if not rows:
        raise SystemExit("Index contains no pages.")

    dims = {int(r[4]) for r in rows}
    if len(dims) != 1:
        raise SystemExit(f"Mixed embedding dimensions in DB: {sorted(dims)}")
    dim = dims.pop()

    matrix = np.empty((len(rows), dim), dtype=np.float32)
    meta = []
    for i, (row_id, path, page, file_sha, _, blob) in enumerate(rows):
        matrix[i] = np.frombuffer(blob, dtype=np.float16, count=dim).astype(np.float32)
        meta.append((row_id, path, int(page), file_sha))

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms != 0)
    return matrix, meta, dim


def load_model(model_id, dim, device):
    import torch
    from sentence_transformers import SentenceTransformer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"Loading {model_id} on {device}, dtype={dtype}, dim={dim} ...", flush=True)
    return SentenceTransformer(
        model_id,
        device=device,
        truncate_dim=dim,
        model_kwargs={"torch_dtype": dtype},
    )


def load_queries(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit("No queries found.")
    return rows


def main():
    p = argparse.ArgumentParser(description="Batch health-pattern search over FolioLense index.")
    p.add_argument("--queries", default="health_queries.jsonl")
    p.add_argument("--db", default="~/.local/share/foliolense/index.sqlite")
    p.add_argument("--output", default="foliolense_results.jsonl")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--query-batch", type=int, default=8)
    args = p.parse_args()

    qrows = load_queries(Path(args.queries))
    conn = connect_db(Path(args.db))

    model_id = get_meta(conn, "model_id") or MODEL_ID
    matrix, meta, dim = load_matrix(conn)

    print(f"Pages in index: {len(meta)}")
    print(f"Queries: {len(qrows)}")
    print(f"Top per query: {args.top}")

    model = load_model(model_id, dim, args.device)

    texts = [q["query"] for q in qrows]
    print("Encoding queries ...", flush=True)
    qmat = model.encode_query(
        texts,
        batch_size=args.query_batch,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    qmat = np.asarray(qmat, dtype=np.float32)
    norms = np.linalg.norm(qmat, axis=1, keepdims=True)
    np.divide(qmat, norms, out=qmat, where=norms != 0)

    out = Path(args.output)
    written = 0

    with out.open("w", encoding="utf-8") as f:
        for qi, qrow in enumerate(qrows):
            scores = matrix @ qmat[qi]
            k = min(args.top, len(scores))
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            for rank, idx in enumerate(top_idx, 1):
                _, path, page, file_sha = meta[int(idx)]
                rec = {
                    "query_id": qrow["query_id"],
                    "cluster_id": qrow.get("cluster_id"),
                    "cluster_label": qrow.get("cluster_label"),
                    "kind": qrow.get("kind", "pattern"),
                    "query": qrow["query"],
                    "rank": rank,
                    "score": round(float(scores[idx]), 6),
                    "source_hash": file_sha,
                    "source_path": path,
                    "page": page,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            print(f"[{qi+1}/{len(qrows)}] {qrow['query_id']} done", flush=True)

    conn.close()
    print(f"Done. Wrote {written} results -> {out}")


if __name__ == "__main__":
    main()
