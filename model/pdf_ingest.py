import os, io, json, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm
from google.cloud import storage
import fitz  # PyMuPDF

from config import (
    GCS_CORPUS_BUCKET,
    WORKDIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    KNOWLEDGE_SOURCE,
    CHUNK_MAX_SIZE,
)

# -------------------------------
# Token counting (real tokenizer if available)
# -------------------------------
try:
    import tiktoken
    _TOK_ENCODER = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(s: str) -> int:
        return len(_TOK_ENCODER.encode(s))
except Exception:
    CHARS_PER_TOKEN = 3  # conservative
    def _count_tokens(s: str) -> int:
        return max(1, len(s) // CHARS_PER_TOKEN)

# -------------------------------
# Helpers
# -------------------------------
def _clean_pdf_text(txt: str) -> str:
    t = txt.replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"(?m)([A-Za-z0-9])-\n([A-Za-z0-9])", r"\1\2", t)  # dehyphenate
    t = re.sub(r"[ \t]{2,}", "  ", t)
    return t

def _slice_by_headers(text: str, headers: List[Tuple[int, int, str]]) -> List[str]:
    if not headers:
        return []
    headers.sort(key=lambda x: x[0])
    chunks = []
    for i, (s, e, title) in enumerate(headers):
        nxt = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[e:nxt].strip()
        chunk = f"{title}\n\n{body}".strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def _detect_source(text: str) -> str:
    fatf_signals = [
        "financial action task force",
        "fatf recommendations",
        "interpretive notes to the fatf recommendations",
        "mutual evaluations (fatf"
    ]
    ofac_signals = [
        "office of foreign assets control",
        "ofac",
        "sanctions program and country information",
        "frequently asked questions"
    ]
    t = text.lower()
    fatf_hits = sum(1 for s in fatf_signals if s in t)
    ofac_hits = sum(1 for s in ofac_signals if s in t)
    if fatf_hits >= ofac_hits:
        return "fatf"
    return "ofac"

# -------------------------------
# OFAC FAQ chunking
# -------------------------------
def _chunk_ofac_faq(text: str) -> List[str]:
    header_start = re.compile(r"(?m)^\s*(?P<num>\d{1,4})\.\s+")
    starts = list(header_start.finditer(text))
    if not starts:
        return []
    headers = []
    for idx, m in enumerate(starts):
        num = m.group("num")
        after_num = m.end()
        s1 = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
        window = text[after_num:s1]
        qm = re.search(r"[?？]", window)
        blank = re.search(r"\n\s*\n", window)
        cap_end = min(len(window), 300)
        if qm:
            heading_end = after_num + qm.end()
        elif blank:
            heading_end = after_num + blank.start()
        else:
            heading_end = after_num + cap_end
        raw_title = text[after_num:heading_end]
        title_clean = re.sub(r"\s+", " ", raw_title).strip()
        title = f"{num}. {title_clean}" if title_clean else f"{num}."
        headers.append((m.start(), heading_end, title))
    return _slice_by_headers(text, headers)

# -------------------------------
# FATF Recommendations & INs
# -------------------------------
def _truncate_at_general_glossary(text: str) -> str:
    token = "GENERAL GLOSSARY"
    idx = text.find(token)
    return text[:idx] if idx != -1 else text

_FATF_IN_BANNER = re.compile(
    r"(?mi)^\s*INTERPRETIVE\s+NOTES\s+TO\s+THE\s+FATF\s+RECOMMENDATIONS\s*$"
)

def _chunk_fatf_recommendations(text: str) -> List[str]:
    text = _truncate_at_general_glossary(text)
    in_banner = _FATF_IN_BANNER.search(text)
    rec_scope = text[: in_banner.start()] if in_banner else text
    anchor_pat = re.compile(
        r"(?mi)^\s*1\.\s*Assessing\s+risks\s+and\s+applying\s+a\s+risk-?based\s+approach\s*\*?\s*$"
    )
    rec_pat = re.compile(
        r"(?m)^\s*(?P<num>[1-9]|[1-3]\d|40)\.\s+(?P<header>[^\n]*?)(?:\s*\*+)?\s*$"
    )
    anchor = anchor_pat.search(rec_scope)
    if not anchor:
        return []
    post = rec_scope[anchor.start():]
    heads = []
    for m in rec_pat.finditer(post):
        header = m.group("header").strip()
        if len(header) < 6:
            continue
        if re.search(r"\.\s*$", header):
            continue
        abs_start = anchor.start() + m.start()
        abs_end = anchor.start() + m.end()
        title = f"{m.group('num')}. {header}"
        heads.append((abs_start, abs_end, title))
    return _slice_by_headers(rec_scope, heads)

def _chunk_fatf_interpretive_notes(text: str) -> List[str]:
    text = _truncate_at_general_glossary(text)
    sec = _FATF_IN_BANNER.search(text)
    if not sec:
        return []
    in_head = re.compile(
        r"(?mi)^\s*INTERPRETIVE\s+NOTE\s+TO\s+RECOMMENDATION\s+(?P<num>[1-9]|[1-3]\d|40)\b.*$"
    )
    start = sec.end()
    tail = text[start:]
    heads = []
    for m in in_head.finditer(tail):
        abs_start = start + m.start()
        abs_end   = start + m.end()
        title     = f"INTERPRETIVE NOTE TO RECOMMENDATION {m.group('num')}"
        heads.append((abs_start, abs_end, title))
    return _slice_by_headers(text, heads)

# -------------------------------
# Fallback chunker
# -------------------------------
def _fallback_chunks(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    n = len(text)
    if size <= 0:
        return chunks
    overlap = max(0, min(overlap, size - 1))
    start = 0
    while start < n:
        end = min(n, start + size)
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks

# -------------------------------
# Safe split to token budget
# -------------------------------
def _split_to_token_budget(title: str, body: str, max_tokens: int) -> List[str]:
    full = f"{title}\n\n{body}".strip() if body else title.strip()
    if _count_tokens(full) <= max_tokens:
        return [full]
    paras = [p.strip() for p in re.split(r"\n{2,}", body or "") if p.strip()]
    prelude = title.strip()
    chunks = []
    cur = [prelude, ""]
    cur_tok = _count_tokens(prelude) + 1
    def flush():
        nonlocal cur, cur_tok
        text = "\n".join(cur).strip()
        if text:
            chunks.append(text)
        cur = [prelude, ""]
        cur_tok = _count_tokens(prelude) + 1
    for para in paras:
        ptok = _count_tokens(para)
        if cur_tok + ptok > max_tokens:
            if cur_tok > _count_tokens(prelude) + 1:
                flush()
            sents = re.split(r"(?<=[.!?])\s+", para)
            buf, buf_tok = [], 0
            for s in sents:
                stok = _count_tokens(s)
                if (_count_tokens(prelude) + 1 + buf_tok + stok) <= max_tokens:
                    buf.append(s)
                    buf_tok += stok
                else:
                    text = "\n".join([prelude, "", " ".join(buf).strip()]).strip()
                    if text:
                        chunks.append(text)
                    buf, buf_tok = [s], stok
            if buf:
                text = "\n".join([prelude, "", " ".join(buf).strip()]).strip()
                if text:
                    chunks.append(text)
        else:
            cur.append(para)
            cur_tok += ptok
    last = "\n".join(cur).strip()
    if last and _count_tokens(last) > _count_tokens(prelude) + 1:
        chunks.append(last)
    out = []
    for c in chunks:
        if _count_tokens(c) <= max_tokens:
            out.append(c)
        else:
            out.extend(_window_split(prelude, c.split("\n\n", 1)[-1], max_tokens))
    return out

def _window_split(title: str, text: str, max_tokens: int) -> List[str]:
    title_tok = _count_tokens(title) + 1
    budget = max(200, max_tokens - title_tok)
    window_chars = max(800, budget * 4)
    parts, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + window_chars)
        piece = f"{title}\n\n{text[i:j].strip()}"
        while _count_tokens(piece) > max_tokens and window_chars > 200:
            window_chars = int(window_chars * 0.9)
            piece = f"{title}\n\n{text[i:i+window_chars].strip()}"
        parts.append(piece)
        i += window_chars
    return parts

# -------------------------------
# GCS utilities
# -------------------------------
def _bucket_and_prefix(gs_or_name: str) -> Tuple[str, str]:
    v = (gs_or_name or "").strip()
    if v.startswith("gs://"):
        b, *rest = v[5:].split("/", 1)
        return b, (rest[0] if rest else "")
    if "/" in v:
        b, p = v.split("/", 1)
        return b, p
    return v, ""

def _resolve_source() -> Tuple[str, str]:
    if KNOWLEDGE_SOURCE.startswith("gs://"):
        return _bucket_and_prefix(KNOWLEDGE_SOURCE)
    base_bucket, base_prefix = _bucket_and_prefix(GCS_CORPUS_BUCKET)
    src_prefix = "/".join([p for p in [base_prefix, KNOWLEDGE_SOURCE.strip("/")] if p])
    return base_bucket, src_prefix

def _extract_text_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text("text") for page in doc)

def _extract_text_textlike(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def list_objects(bucket_name: str, prefix: str) -> List[str]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return [b.name for b in blobs if not b.name.endswith("/")]

def download_blob(bucket_name: str, blob_name: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.download_to_filename(dest_path)

def extract_text_generic(local_path: str, blob_name: str) -> str:
    name = blob_name.lower()
    if name.endswith(".pdf"):
        return _extract_text_pdf(local_path)
    elif name.endswith((".txt", ".json", ".jsonl", ".md", ".csv")):
        return _extract_text_textlike(local_path)
    else:
        try:
            return _extract_text_textlike(local_path)
        except Exception:
            return ""

# -------------------------------
# Main chunk orchestrator
# -------------------------------
def chunk_text(txt: str, size: int = 1200, overlap: int = 150, source: Optional[str] = None) -> List[str]:
    cleaned = _clean_pdf_text(txt)
    src = (source or _detect_source(cleaned)).lower()
    if src == "fatf":
        chunks = _chunk_fatf_recommendations(cleaned) + _chunk_fatf_interpretive_notes(cleaned)
        if not chunks:
            chunks = _fallback_chunks(cleaned, size=size, overlap=overlap)
    else:
        chunks = _chunk_ofac_faq(cleaned) or _fallback_chunks(cleaned, size=size, overlap=overlap)
    safe_chunks = []
    for ch in chunks:
        if "\n" in ch:
            first_line, rest = ch.split("\n", 1)
        else:
            first_line, rest = ch, ""
        safe_chunks.extend(_split_to_token_budget(first_line.strip(), rest.strip(), CHUNK_MAX_SIZE))
    really_safe = []
    for c in safe_chunks:
        if _count_tokens(c) <= CHUNK_MAX_SIZE:
            really_safe.append(c)
        else:
            title = c.split("\n", 1)[0]
            body  = c[len(title):].lstrip("\n")
            really_safe.extend(_split_to_token_budget(title, body, CHUNK_MAX_SIZE))
    return really_safe

# -------------------------------
# Driver
# -------------------------------
def main():
    os.makedirs(WORKDIR, exist_ok=True)
    bucket_name, src_prefix = _resolve_source()
    keys = list_objects(bucket_name, src_prefix)
    if not keys:
        print(f"No objects found under gs://{bucket_name}/{src_prefix}")
        return
    corpus_jsonl = os.path.join(WORKDIR, "corpus_chunks.jsonl")
    with open(corpus_jsonl, "w", encoding="utf-8") as out:
        for key in tqdm(keys, desc="Downloading + extracting"):
            local_path = os.path.join(WORKDIR, key.replace("/", "_"))
            download_blob(bucket_name, key, local_path)
            text = extract_text_generic(local_path, key)
            if not text.strip():
                continue
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, ch in enumerate(chunks):
                out.write(json.dumps({"doc": key, "chunk_id": idx, "text": ch}, ensure_ascii=False) + "\n")
    print(f"✅ Wrote chunked corpus to {corpus_jsonl}")

if __name__ == "__main__":
    main()
