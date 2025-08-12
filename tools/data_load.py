# tools/data_load.py
import argparse
import mimetypes
from pathlib import Path
from typing import Tuple, Iterable

from google.cloud import storage

import config as C

DATA_DIR = Path(getattr(C, "LOCAL_DATA", "data"))
KNOWLEDGE_SOURCE = getattr(C, "KNOWLEDGE_SOURCE", "knowledge_source")
GCS_CORPUS_BUCKET = getattr(C, "GCS_CORPUS_BUCKET", None)  # used if KNOWLEDGE_SOURCE is not a full gs://

def _bucket_and_prefix(gs_or_name: str) -> Tuple[str, str]:
    """
    Accepts:
      - 'gs://bucket/a/b'  -> ('bucket', 'a/b')
      - 'bucket/a/b'       -> ('bucket', 'a/b')
      - 'bucket'           -> ('bucket', '')
    """
    v = (gs_or_name or "").strip()
    if v.startswith("gs://"):
        b, *rest = v[5:].split("/", 1)
        return b, (rest[0] if rest else "")
    if "/" in v:
        b, p = v.split("/", 1)
        return b, p
    return v, ""

def _resolve_target() -> Tuple[str, str]:
    """
    Determine (bucket, prefix) for upload target using KNOWLEDGE_SOURCE and (optionally) GCS_CORPUS_BUCKET.
    - If KNOWLEDGE_SOURCE is full gs://... -> use it as-is.
    - Else require GCS_CORPUS_BUCKET and append KNOWLEDGE_SOURCE as sub-prefix.
    """
    if KNOWLEDGE_SOURCE.startswith("gs://") or KNOWLEDGE_SOURCE.split("/", 1)[0].endswith(".appspot.com"):
        return _bucket_and_prefix(KNOWLEDGE_SOURCE)
    if not GCS_CORPUS_BUCKET:
        raise SystemExit("❌ KNOWLEDGE_SOURCE is not a gs:// URI and GCS_CORPUS_BUCKET is not set in config.py")
    bkt, base = _bucket_and_prefix(GCS_CORPUS_BUCKET)
    sub = KNOWLEDGE_SOURCE.strip("/")

    prefix = "/".join([p for p in [base, sub] if p])
    return bkt, prefix

def _iter_pdfs(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob("*.pdf")
        yield from root.rglob("*.PDF")
    else:
        yield from root.glob("*.pdf")
        yield from root.glob("*.PDF")

def upload_pdfs(recursive: bool = True, overwrite: bool = False) -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"❌ DATA_DIR does not exist: {DATA_DIR}")

    bucket_name, prefix = _resolve_target()
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    count = 0
    for pdf in _iter_pdfs(DATA_DIR, recursive=recursive):
        if not pdf.is_file():
            continue
        rel = pdf.relative_to(DATA_DIR).as_posix()
        object_path = "/".join([p for p in [prefix, rel] if p])
        blob = bucket.blob(object_path)

        if blob.exists() and not overwrite:
            print(f"↩︎  Skipping (exists): gs://{bucket_name}/{object_path}")
            continue

        ctype, _ = mimetypes.guess_type(pdf.name)
        if not ctype:
            ctype = "application/pdf"

        blob.upload_from_filename(str(pdf), content_type=ctype)
        print(f"☁️  Uploaded: gs://{bucket_name}/{object_path}")
        count += 1

    if count == 0:
        print("ℹ️  No new PDFs uploaded.")
    else:
        print(f"✅ Done. Uploaded {count} PDF(s).")

def main():
    ap = argparse.ArgumentParser(description="Upload local PDFs to KNOWLEDGE_SOURCE in GCS.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders of DATA_DIR")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing GCS objects")
    args = ap.parse_args()
    upload_pdfs(recursive=args.recursive, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
