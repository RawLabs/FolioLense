#!/usr/bin/env python3
"""
foliolense_crawl.py
Resumable local visual-document indexer/searcher for:
  tomaarsen/Qwen3-VL-Embedding-2B-vdr

Designed for a single NVIDIA GPU with limited VRAM:
- recursive crawl
- PDFs rendered one page at a time
- batch size 1
- SQLite checkpointing after every page
- skips unchanged files on later runs
- optional image-file indexing
- compact float16 embedding storage
- local cosine search with NumPy (no vector DB required)

Examples:
  python3 foliolense_crawl.py crawl ~/Documents --db ~/foliolense-index.sqlite
  python3 foliolense_crawl.py query "4-hour shifts non-consecutive days" --db ~/foliolense-index.sqlite
  python3 foliolense_crawl.py stats --db ~/foliolense-index.sqlite

Install:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U "sentence-transformers[image]" pymupdf pillow numpy

You also need a CUDA-enabled PyTorch build for GPU inference.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sqlite3
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

MODEL_ID = "tomaarsen/Qwen3-VL-Embedding-2B-vdr"
DEFAULT_DIM = 512
DEFAULT_DPI = 144

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    indexed_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'complete',
    error TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    page INTEGER NOT NULL,
    file_sha256 TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    indexed_at REAL NOT NULL,
    UNIQUE(path, page)
);

CREATE INDEX IF NOT EXISTS idx_pages_path ON pages(path);
"""


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def connect_db(path: Path) -> sqlite3.Connection:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def configure_index(conn: sqlite3.Connection, model_id: str, dim: int, dpi: int) -> None:
    old_model = get_meta(conn, "model_id")
    old_dim = get_meta(conn, "dim")

    if old_model and old_model != model_id:
        raise SystemExit(
            f"Index already uses model {old_model!r}; refusing to mix with {model_id!r}."
        )
    if old_dim and int(old_dim) != dim:
        raise SystemExit(
            f"Index already uses {old_dim}-dim embeddings; requested {dim}. "
            "Use a different --db or keep the same --dim."
        )

    set_meta(conn, "model_id", model_id)
    set_meta(conn, "dim", str(dim))
    set_meta(conn, "dpi", str(dpi))
    set_meta(conn, "embedding_dtype", "float16")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def unchanged(conn: sqlite3.Connection, path: Path, st: os.stat_result) -> bool:
    row = conn.execute(
        "SELECT size, mtime_ns, status FROM files WHERE path=?",
        (str(path),),
    ).fetchone()
    return bool(
        row
        and row[0] == st.st_size
        and row[1] == st.st_mtime_ns
        and row[2] == "complete"
    )


def discover(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in PDF_EXTS | IMAGE_EXTS:
            yield root.resolve()
        return

    for dirpath, dirnames, filenames in os.walk(root):
        # Avoid crawling hidden/cache folders by default.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d not in {"node_modules", "__pycache__", ".git"}
        ]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in PDF_EXTS | IMAGE_EXTS:
                yield p.resolve()


def load_model(model_id: str, dim: int, device: str):
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing model dependencies. Run:\n"
            '  pip install -U "sentence-transformers[image]"'
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but PyTorch cannot see the NVIDIA GPU.\n"
            "Check with:\n"
            '  python3 -c "import torch; print(torch.cuda.is_available())"'
        )

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    eprint(f"Loading {model_id} on {device}, dtype={dtype}, dim={dim} ...")
    model = SentenceTransformer(
        model_id,
        device=device,
        truncate_dim=dim,
        model_kwargs={"torch_dtype": dtype},
    )
    return model


def render_pdf_page(page, dpi: int):
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img


def open_image_file(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def embed_image(model, image) -> np.ndarray:
    """
    Returns one L2-normalized document vector.
    Batch size stays at 1 intentionally for 8 GB-class GPUs.
    """
    emb = model.encode_document(
        [image],
        batch_size=1,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    emb = np.asarray(emb, dtype=np.float32)

    # The model includes normalization, but normalize again defensively before storage.
    norm = float(np.linalg.norm(emb))
    if norm > 0:
        emb /= norm
    return emb


def store_page(
    conn: sqlite3.Connection,
    *,
    path: Path,
    page_num: int,
    file_sha: str,
    image,
    embedding: np.ndarray,
) -> None:
    packed = embedding.astype(np.float16).tobytes()
    conn.execute(
        """
        INSERT INTO pages(path,page,file_sha256,width,height,dim,embedding,indexed_at)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(path,page) DO UPDATE SET
            file_sha256=excluded.file_sha256,
            width=excluded.width,
            height=excluded.height,
            dim=excluded.dim,
            embedding=excluded.embedding,
            indexed_at=excluded.indexed_at
        """,
        (
            str(path),
            page_num,
            file_sha,
            int(image.width),
            int(image.height),
            int(embedding.shape[0]),
            packed,
            time.time(),
        ),
    )
    # Checkpoint every page so Ctrl+C/reboot loses at most the current page.
    conn.commit()


def mark_file(
    conn: sqlite3.Connection,
    path: Path,
    st: os.stat_result,
    sha: str,
    status: str,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO files(path,size,mtime_ns,sha256,indexed_at,status,error)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
            size=excluded.size,
            mtime_ns=excluded.mtime_ns,
            sha256=excluded.sha256,
            indexed_at=excluded.indexed_at,
            status=excluded.status,
            error=excluded.error
        """,
        (str(path), st.st_size, st.st_mtime_ns, sha, time.time(), status, error),
    )
    conn.commit()


def index_pdf(conn, model, path: Path, file_sha: str, dpi: int) -> int:
    import fitz  # PyMuPDF

    count = 0
    with fitz.open(path) as doc:
        total = len(doc)

        # If this same file hash was partially indexed, resume at missing pages.
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT page FROM pages WHERE path=? AND file_sha256=?",
                (str(path), file_sha),
            )
        }

        for zero_idx in range(total):
            page_num = zero_idx + 1
            if page_num in existing:
                continue

            img = render_pdf_page(doc[zero_idx], dpi)
            emb = embed_image(model, img)
            store_page(
                conn,
                path=path,
                page_num=page_num,
                file_sha=file_sha,
                image=img,
                embedding=emb,
            )
            count += 1
            print(f"  page {page_num}/{total}", flush=True)
            del img, emb

    return count


def index_image(conn, model, path: Path, file_sha: str) -> int:
    img = open_image_file(path)
    emb = embed_image(model, img)
    store_page(
        conn,
        path=path,
        page_num=1,
        file_sha=file_sha,
        image=img,
        embedding=emb,
    )
    return 1


def prune_missing(conn: sqlite3.Connection, root: Path) -> int:
    """Remove missing paths only inside the accessible folder being updated."""
    removed = 0
    paths = conn.execute("SELECT path FROM files UNION SELECT path FROM pages").fetchall()
    for (stored,) in paths:
        path = Path(stored)
        if path != root and root not in path.parents:
            continue
        try:
            path.stat()
        except FileNotFoundError:
            with conn:
                conn.execute("DELETE FROM pages WHERE path=?", (stored,))
                conn.execute("DELETE FROM files WHERE path=?", (stored,))
            removed += 1
        except OSError:
            # Permission or storage errors are not evidence of deletion.
            continue
    return removed


def open_result(path: str, page: int) -> None:
    document = Path(path)
    if not document.is_file():
        print("That file is no longer available. Update its folder with menu option 1.")
        return
    opener = shutil.which("xdg-open")
    if not opener:
        print(f"No desktop file opener found. Open this file manually: {path}")
        return
    try:
        result = subprocess.run(
            [opener, str(document.resolve())], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        print(f"Could not open the document. Open it manually: {path}")
        return
    if result.returncode:
        print(f"Could not open the document. Open it manually: {path}")
    else:
        print(f"Opened {document.name}. Go to page {page} in your viewer.")


def choose_result(results) -> None:
    while True:
        try:
            choice = input("Open result number (Enter returns to the menu): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not choice:
            return
        if not choice.isascii() or not choice.isdigit() or not 1 <= int(choice) <= len(results):
            print(f"Enter a result number from 1 to {len(results)}, or press Enter.")
            continue
        _, path, page = results[int(choice) - 1]
        open_result(path, page)


def crawl(args) -> None:
    root = Path(args.root).expanduser().resolve()
    db = Path(args.db)
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    conn = connect_db(db)
    configure_index(conn, args.model, args.dim, args.dpi)
    removed = prune_missing(conn, root)
    if removed:
        print(f"Removed {removed} missing document(s) from this folder’s search index.")

    files = list(discover(root))
    print(f"Found {len(files)} supported files under {root}")

    if not files:
        print("No supported PDFs or images found in this location.")
        conn.close()
        return
    model = load_model(args.model, args.dim, args.device)

    new_pages = 0
    skipped = 0
    failures = 0

    try:
        for i, path in enumerate(files, 1):
            try:
                st = path.stat()
                if unchanged(conn, path, st) and not args.force:
                    skipped += 1
                    print(f"[{i}/{len(files)}] skip {path}")
                    continue

                print(f"[{i}/{len(files)}] index {path}", flush=True)
                file_sha = sha256_file(path)

                # If file content changed, remove old page rows first.
                prior = conn.execute(
                    "SELECT sha256 FROM files WHERE path=?", (str(path),)
                ).fetchone()
                if prior and prior[0] != file_sha:
                    conn.execute("DELETE FROM pages WHERE path=?", (str(path),))
                    conn.commit()

                mark_file(conn, path, st, file_sha, "indexing")

                if path.suffix.lower() == ".pdf":
                    added = index_pdf(conn, model, path, file_sha, args.dpi)
                else:
                    # Re-index a standalone image as a single document page.
                    conn.execute("DELETE FROM pages WHERE path=?", (str(path),))
                    conn.commit()
                    added = index_image(conn, model, path, file_sha)

                mark_file(conn, path, st, file_sha, "complete")
                new_pages += added

            except KeyboardInterrupt:
                print("\nInterrupted. Completed pages are already checkpointed.")
                raise
            except Exception as exc:
                failures += 1
                try:
                    st = path.stat()
                    prior_sha = conn.execute(
                        "SELECT sha256 FROM files WHERE path=?", (str(path),)
                    ).fetchone()
                    sha = prior_sha[0] if prior_sha else ""
                    mark_file(conn, path, st, sha, "error", repr(exc))
                except Exception:
                    pass
                eprint(f"ERROR: {path}: {exc}")

    except KeyboardInterrupt:
        pass
    finally:
        total_pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        conn.close()

    print(
        f"\nDone. New pages: {new_pages} | skipped files: {skipped} "
        f"| failures: {failures} | indexed pages total: {total_pages}"
    )


def load_matrix(conn: sqlite3.Connection, *, available_only: bool = False):
    rows = conn.execute(
        "SELECT id,path,page,dim,embedding FROM pages ORDER BY id"
    ).fetchall()
    if available_only:
        rows = [row for row in rows if Path(row[1]).is_file()]
    if not rows:
        raise SystemExit("No searchable pages yet. Add a PDF or image using menu option 1, or run the crawl command.")

    dims = {int(r[3]) for r in rows}
    if len(dims) != 1:
        raise SystemExit(f"Mixed embedding dimensions in DB: {sorted(dims)}")
    dim = dims.pop()

    # float16 on disk -> float32 for fast/stable cosine dot products.
    matrix = np.empty((len(rows), dim), dtype=np.float32)
    meta = []
    for i, (row_id, path, page, _, blob) in enumerate(rows):
        matrix[i] = np.frombuffer(blob, dtype=np.float16, count=dim).astype(np.float32)
        meta.append((row_id, path, page))

    # Re-normalize after float16 round-trip.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms != 0)
    return matrix, meta, dim


def query(args) -> None:
    db = Path(args.db)
    conn = connect_db(db)

    model_id = get_meta(conn, "model_id") or args.model
    dim = int(get_meta(conn, "dim") or args.dim)

    matrix, meta, dim = load_matrix(conn, available_only=True)
    model = load_model(model_id, dim, args.device)

    q = model.encode_query(
        [args.text],
        batch_size=1,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    q = np.asarray(q, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm > 0:
        q /= q_norm

    scores = matrix @ q
    k = min(args.top, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    print(f"\nSearch: {args.text}")
    print("Results: relevance score | page number | file path")
    print("Higher scores mean closer matches, not percentages.\n")
    for rank, idx in enumerate(top_idx, 1):
        _, path, page = meta[int(idx)]
        print(f"{rank:2d}. {scores[idx]:.4f}  page {page:4d}  {path}")

    conn.close()
    if getattr(args, "interactive", False):
        choose_result([meta[int(idx)] for idx in top_idx])


def stats(args) -> None:
    conn = connect_db(Path(args.db))
    file_count = conn.execute(
        "SELECT COUNT(*) FROM files WHERE status='complete'"
    ).fetchone()[0]
    error_count = conn.execute(
        "SELECT COUNT(*) FROM files WHERE status='error'"
    ).fetchone()[0]
    page_count = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    blob_bytes = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(embedding)),0) FROM pages"
    ).fetchone()[0]

    print(f"Model:       {get_meta(conn, 'model_id')}")
    print(f"Dimensions:  {get_meta(conn, 'dim')}")
    print(f"Render DPI:  {get_meta(conn, 'dpi')}")
    print(f"Files:       {file_count}")
    print(f"Pages:       {page_count}")
    print(f"Errors:      {error_count}")
    print(f"Vectors:     {blob_bytes / (1024**2):.1f} MiB")

    if error_count:
        print("\nErrors:")
        for path, err in conn.execute(
            "SELECT path,error FROM files WHERE status='error' ORDER BY path"
        ):
            print(f"- {path}\n  {err}")
    conn.close()


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Enter a whole number greater than zero.")
    return number


def build_parser():
    p = argparse.ArgumentParser(
        description="Resumable visual-document crawler/searcher using Qwen3-VL-Embedding-2B-vdr."
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("crawl", help="Recursively index PDFs and images.")
    c.add_argument("root", help="File or directory to crawl.")
    c.add_argument("--db", default="foliolense-index.sqlite", help="SQLite index path.")
    c.add_argument("--model", default=MODEL_ID)
    c.add_argument("--dim", type=int, default=DEFAULT_DIM,
                   choices=[64, 128, 256, 512, 1024, 1536, 2048])
    c.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    c.add_argument("--device", default="cuda")
    c.add_argument("--force", action="store_true",
                   help="Re-check/re-index files even if size+mtime are unchanged.")
    c.set_defaults(func=crawl)

    q = sub.add_parser("query", help="Search the local page index.")
    q.add_argument("text", help="Natural-language query.")
    q.add_argument("--db", default="foliolense-index.sqlite")
    q.add_argument("--model", default=MODEL_ID)
    q.add_argument("--dim", type=int, default=DEFAULT_DIM)
    q.add_argument("--device", default="cuda")
    q.add_argument("--top", type=positive_int, default=15)
    q.add_argument("--interactive", action="store_true",
                   help="Choose search results to open in your desktop viewer.")
    q.set_defaults(func=query)

    s = sub.add_parser("stats", help="Show index statistics/errors.")
    s.add_argument("--db", default="foliolense-index.sqlite")
    s.set_defaults(func=stats)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
