import os, io, json, math
from typing import List
from tqdm import tqdm
from google.cloud import storage
import fitz  # PyMuPDF
from config import GCS_CORPUS_BUCKET, WORKDIR, CHUNK_SIZE, CHUNK_OVERLAP

def list_pdfs_in_bucket(bucket_name: str) -> List[str]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs())
    return [b.name for b in blobs if b.name.lower().endswith(".pdf")]

def download_blob(bucket_name: str, blob_name: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(dest_path)

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)

import re
from typing import List

def chunk_text(txt: str, size: int = 1200, overlap: int = 150) -> List[str]:
    """
    Split text into FAQ chunks where a heading starts with N., NN., NNN., or NNNN.
    The heading (question) must end with '?'. Each chunk includes the question and
    its answer text up to (but not including) the next numbered question.

    Falls back to size/overlap chunking if no FAQ pattern is found.
    """
    # Normalize a bit (PDFs can be messy)
    cleaned = txt.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)  # trim trailing spaces before newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # collapse huge gaps

    # Regex:
    # - start of line
    # - 1-4 digits then a dot
    # - some whitespace
    # - lazily grab any text ending with '?'
    # - capture answer body until the next numbered question or end of text
    faq_pattern = re.compile(
        r"(?m)^\s*(\d{1,4})\.\s+(?P<question>.+?\?)\s*(?P<body>.*?)(?=^\s*\d{1,4}\.\s+.+?\?|\Z)",
        re.DOTALL
    )

    chunks = []
    for m in faq_pattern.finditer(cleaned):
        q = m.group("question").strip()
        body = m.group("body").strip()
        chunk = f"{q}\n\n{body}".strip()
        if chunk:  # avoid empties
            chunks.append(chunk)

    if chunks:
        return chunks  # FAQ-based chunks found — we're done

    # Fallback: size/overlap chunking if no numbered FAQs were detected
    fallback_chunks = []
    start = 0
    n = len(cleaned)
    # guard against goofy params
    overlap = max(0, min(overlap, size - 1)) if size > 0 else 0
    while size > 0 and start < n:
        end = min(n, start + size)
        fallback_chunks.append(cleaned[start:end])
        # advance with overlap
        start = end - overlap if end - overlap > start else end
    return fallback_chunks


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    pdf_keys = list_pdfs_in_bucket(GCS_CORPUS_BUCKET)
    if not pdf_keys:
        print("No PDFs found in bucket.")
        return

    corpus_jsonl = os.path.join(WORKDIR, "corpus_chunks.jsonl")
    with open(corpus_jsonl, "w", encoding="utf-8") as out:
        for key in tqdm(pdf_keys, desc="Downloading + extracting"):
            local_pdf = os.path.join(WORKDIR, key.replace("/", "_"))
            download_blob(GCS_CORPUS_BUCKET, key, local_pdf)
            text = extract_text_from_pdf(local_pdf)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, ch in enumerate(chunks):
                out.write(json.dumps({"doc": key, "chunk_id": idx, "text": ch}) + "\n")

    print(f"✅ Wrote chunked corpus to {corpus_jsonl}")

if __name__ == "__main__":
    main()
