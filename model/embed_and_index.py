# model/embed_and_index.py
import os, json
from pathlib import Path
from typing import Iterable, List, Tuple
import math
import numpy as np
from vertexai import init as vertexai_init
from vertexai.language_models import TextEmbeddingModel

# Try both faiss-cpu and faiss (conda vs pip)
try:
    import faiss                      # conda/Apple Silicon often uses this name
except ImportError:                   # pragma: no cover
    import faiss_cpu as faiss

from config import (
    PROJECT_ID, LOCATION, EMBEDDING_MODEL, WORKDIR,
    # Optional (provide defaults if not present)
)

# -------- optional config with sane defaults --------
EMBEDDING_OUT_VECS   = getattr(__import__("config"), "EMBEDDING_OUT_VECS", "embeddings.npy")
EMBEDDING_OUT_META   = getattr(__import__("config"), "EMBEDDING_OUT_META", "embeddings_meta.jsonl")
FAISS_INDEX_NAME     = getattr(__import__("config"), "FAISS_INDEX_NAME", "faiss.index")
BATCH_SIZE           = getattr(__import__("config"), "EMBED_BATCH_SIZE", 32)
UPLOAD_INDEX_TO_GCS  = getattr(__import__("config"), "UPLOAD_INDEX_TO_GCS", False)
GCS_CORPUS_BUCKET    = getattr(__import__("config"), "GCS_CORPUS_BUCKET", "")
INDEX_PREFIX         = getattr(__import__("config"), "INDEX_PREFIX", "index")

# Vertex cap is ~20k per request; keep margin because tokenizers differ
REQUEST_TOKEN_BUDGET = getattr(__import__("config"), "EMBED_REQUEST_TOKEN_BUDGET", 18000)
# ----------------------------------------------------

def _bucket_and_prefix(gs_or_name: str) -> Tuple[str, str]:
    v = (gs_or_name or "").strip()
    if not v:
        return "", ""
    if v.startswith("gs://"):
        b, *rest = v[5:].split("/", 1)
        return b, (rest[0] if rest else "")
    return v, ""

def _upload_file_to_gcs(local_path: Path, dest_name: str) -> str:
    from google.cloud import storage
    bucket_name, base = _bucket_and_prefix(GCS_CORPUS_BUCKET)
    if not bucket_name:
        raise RuntimeError("GCS_CORPUS_BUCKET not set for upload.")
    object_path = "/".join([p for p in [base, INDEX_PREFIX, dest_name] if p])
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{object_path}"

def load_chunks_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
def estimate_tokens(s: str) -> int:
    # Try tiktoken if present; fall back to a simple heuristic
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    except Exception:
        words = len(s.split()); chars = len(s)
        return max(math.ceil(words * 1.33), math.ceil(chars / 4))


def _bucket_and_prefix(gs_or_name: str) -> Tuple[str, str]:
    v = (gs_or_name or "").strip()
    if not v:
        return "", ""
    if v.startswith("gs://"):
        b, *rest = v[5:].split("/", 1)
        return b, (rest[0] if rest else "")
    return v, ""

def _upload_file_to_gcs(local_path: Path, dest_name: str) -> str:
    from google.cloud import storage
    bucket_name, base = _bucket_and_prefix(GCS_CORPUS_BUCKET)
    if not bucket_name:
        raise RuntimeError("GCS_CORPUS_BUCKET not set for upload.")
    object_path = "/".join([p for p in [base, INDEX_PREFIX, dest_name] if p])
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{object_path}"

def main():
    # Init Vertex + model
    vertexai_init(project=PROJECT_ID, location=LOCATION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    workdir = Path(WORKDIR)
    src_corpus = workdir / "corpus_chunks.jsonl"
    out_vecs   = workdir / EMBEDDING_OUT_VECS
    out_meta   = workdir / EMBEDDING_OUT_META
    out_faiss  = workdir / FAISS_INDEX_NAME

    if not src_corpus.exists():
        raise SystemExit(f"❌ Missing source corpus: {src_corpus}")

    embeddings: List[List[float]] = []
    meta: List[dict] = []

    batch_texts, batch_meta = [], []
    batch_tokens = 0

    def flush():
        nonlocal batch_texts, batch_meta, batch_tokens
        if not batch_texts:
            return
        res = model.get_embeddings(batch_texts)
        for m, emb in zip(batch_meta, res):
            embeddings.append(emb.values)
            meta.append(m)
        batch_texts, batch_meta, batch_tokens = [], [], 0

    for rec in load_chunks_jsonl(src_corpus):
        txt = rec.get("text", "") or ""
        tks = estimate_tokens(txt)

        # If a single item is somehow > budget, embed it alone
        if tks > REQUEST_TOKEN_BUDGET:
            if batch_texts:
                flush()
            # still safe: one-per-request
            res = model.get_embeddings([txt])
            embeddings.append(res[0].values)
            meta.append({"doc": rec.get("doc"), "chunk_id": rec.get("chunk_id")})
            continue

        # If adding this would overflow the request, flush first
        if batch_tokens + tks > REQUEST_TOKEN_BUDGET or len(batch_texts) >= BATCH_SIZE:
            flush()

        batch_texts.append(txt)
        batch_meta.append({"doc": rec.get("doc"), "chunk_id": rec.get("chunk_id")})
        batch_tokens += tks

    flush()

    if not embeddings:
        raise SystemExit("❌ No embeddings generated (empty corpus?).")

    vecs = np.asarray(embeddings, dtype="float32")
    np.save(out_vecs, vecs)
    with out_meta.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # FAISS index: normalize + inner product
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(out_faiss))

    print(f"✅ Saved {vecs.shape[0]} embeddings")
    print(f"   vectors: {out_vecs}")
    print(f"   meta:    {out_meta}")
    print(f"   index:   {out_faiss}")

    upload_index_files()

    # Optional upload to GCS
    if UPLOAD_INDEX_TO_GCS:
        try:
            uri1 = _upload_file_to_gcs(out_vecs, out_vecs.name)
            uri2 = _upload_file_to_gcs(out_meta, out_meta.name)
            uri3 = _upload_file_to_gcs(out_faiss, out_faiss.name)
            print("☁️  Uploaded artifacts:")
            print("   ", uri1)
            print("   ", uri2)
            print("   ", uri3)
        except Exception as e:
            print("⚠️  Upload skipped/failed:", e)

import os
from pathlib import Path
from google.cloud import storage
from config import GCS_CORPUS_BUCKET, WORKDIR, INDEX_PREFIX

def upload_index_files():
    """Uploads embeddings, faiss index, meta, and corpus_chunks.jsonl to the GCS index folder."""
    files_to_upload = [
        "embeddings.npy",
        "faiss.index",
        "embeddings_meta.jsonl",
        "corpus_chunks.jsonl"
    ]

    # Resolve bucket and optional base prefix from GCS_CORPUS_BUCKET
    def _bucket_and_prefix(gs_or_name: str):
        v = gs_or_name.strip()
        if v.startswith("gs://"):
            parts = v[5:].split("/", 1)
            return parts[0], (parts[1] if len(parts) > 1 else "")
        if "/" in v:
            b, p = v.split("/", 1)
            return b, p
        return v, ""

    bucket_name, base_prefix = _bucket_and_prefix(GCS_CORPUS_BUCKET)

    client = storage.Client()

    for fname in files_to_upload:
        local_path = Path(WORKDIR) / fname
        if not local_path.exists():
            print(f"⚠️  File not found, skipping: {local_path}")
            continue

        object_path = "/".join(
            [p for p in [base_prefix, INDEX_PREFIX, fname] if p]
        )
        blob = client.bucket(bucket_name).blob(object_path)
        blob.upload_from_filename(str(local_path))
        print(f"✅ Uploaded {fname} → gs://{bucket_name}/{object_path}")


if __name__ == "__main__":
    main()
