#!/usr/bin/env python3
"""
Vertex RAG Pipeline (starter, low-cost-first)

What this gives you (ready to run / adapt):
- Load PDFs from a folder
- Extract text + basic style-aware metadata (headings-ish heuristics)
- Clean + chunk text into variable-sized, semantically sensible chunks
- Write chunks to JSONL with metadata
- Upload JSONL to your GCS bucket
- (Stubs) Create embeddings + vector index on Vertex AI Vector Search
- (Stubs) Fine-tune Gemini (requires supervised prompt/response JSONL)
- (Stubs) Validation harness: compare baseline vs tuned model on your prompts
- (Stubs) Optional distillation to a smaller open-source model to cut inference costs
- (Stubs) Deploy tuned model to a Vertex endpoint

IMPORTANT NOTES
- Gemini fine-tuning generally expects supervised examples (prompt/response pairs).
  If you only have raw domain text, prefer RAG (embeddings + retrieval) first.
- Keep costs low by: small evaluation sets, minimal epochs, small machine types,
  and doing as much preprocessing locally as possible.
- Adapt the STUBS to your environment; the Vertex SDK evolves quickly.

Usage (example):
    python vertex_rag_pipeline.py \
      --project YOUR_PROJECT_ID \
      --location europe-west4 \
      --gcs_bucket your-bucket-name \
      --pdf_dir ./pdfs \
      --output_prefix corpora/industry_risks \
      --max_chunk_tokens 900 \
      --overlap_tokens 150

Dependencies (install locally):
    pip install google-cloud-aiplatform google-cloud-storage pdfplumber tiktoken tqdm
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import pdfplumber
from tqdm import tqdm

# Tokenizer for chunk sizing (OpenAI tiktoken works fine for rough counts; replace if desired)
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(s: str) -> int:
        return len(_ENC.encode(s))
except Exception:
    # Fallback: crude whitespace token count
    def count_tokens(s: str) -> int:
        return max(1, len(s.split()))

# ===============
# Data structures
# ===============

@dataclass
class Chunk:
    id: str
    text: str
    source_pdf: str
    page_start: int
    page_end: int
    heading: Optional[str] = None
    style: Optional[Dict] = None
    industry: Optional[str] = None
    risk_factor: Optional[str] = None


# =========================
# PDF extraction & cleaning
# =========================

HEADING_REGEX = re.compile(r"^(?:[A-Z][A-Z \d\-:]{4,}|[A-Z][^a-z]{6,})$")

def extract_pdf_text_with_layout(pdf_path: str) -> List[Dict]:
    """
    Extract text from a PDF with basic layout cues.
    Returns a list of dicts: {page_num, blocks:[{text, bbox, fontname, size}...]}
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                objs = page.extract_words(use_text_flow=True, keep_blank_chars=False, extra_attrs=[
                    "fontname", "size", "x0", "x1", "top", "bottom"
                ])
            except Exception:
                objs = []

            # Group by "line" using the 'top' coordinate (rough heuristic)
            lines = {}
            for w in objs or []:
                key = round(w.get("top", 0), 1)
                lines.setdefault(key, []).append(w)

            blocks = []
            for key in sorted(lines.keys()):
                line_words = sorted(lines[key], key=lambda w: w.get("x0", 0))
                text = " ".join([w["text"] for w in line_words]).strip()
                if not text:
                    continue
                # Heuristic for heading-ish: bigger size OR all-caps
                sizes = [w.get("size") for w in line_words if w.get("size")]
                avg_size = sum(sizes)/len(sizes) if sizes else None
                fontnames = list({w.get("fontname") for w in line_words if w.get("fontname")})
                is_allcaps = bool(HEADING_REGEX.match(text)) or text.isupper()
                blocks.append({
                    "text": text,
                    "avg_size": avg_size,
                    "fontnames": fontnames,
                    "is_heading_like": is_allcaps or (avg_size and avg_size >= 12),  # tweak threshold
                })
            pages.append({"page_num": i, "blocks": blocks})
    return pages

def normalize_whitespace(s: str) -> str:
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def pages_to_paragraphs(pages: List[Dict]) -> List[Tuple[int, str, Optional[str], Dict]]:
    """
    Convert page blocks into paragraphs with an optional "current heading".
    Returns list of tuples: (page_num, paragraph_text, heading, style)
    """
    paragraphs = []
    current_heading = None
    for p in pages:
        for b in p["blocks"]:
            text = normalize_whitespace(b["text"])
            if not text:
                continue
            if b.get("is_heading_like"):
                current_heading = text[:120]
                continue
            # Treat each line-like block as paragraph-ish; merge short lines
            paragraphs.append(
                (p["page_num"], text, current_heading, {"avg_size": b.get("avg_size"), "fontnames": b.get("fontnames")})
            )
    # Merge tiny paragraphs into neighbors
    merged = []
    buf = []
    buf_tokens = 0
    for (pg, txt, head, style) in paragraphs:
        tks = count_tokens(txt)
        if buf_tokens + tks < 60 and (buf and head == buf[-1][2]):  # small lines join
            buf.append((pg, txt, head, style))
            buf_tokens += tks
        else:
            if buf:
                merged_txt = " ".join(x[1] for x in buf)
                merged.append((buf[0][0], merged_txt, buf[0][2], buf[0][3]))
            buf = [(pg, txt, head, style)]
            buf_tokens = tks
    if buf:
        merged_txt = " ".join(x[1] for x in buf)
        merged.append((buf[0][0], merged_txt, buf[0][2], buf[0][3]))
    return merged

# ============
# Chunking
# ============

def chunk_paragraphs(
    paragraphs: List[Tuple[int, str, Optional[str], Dict]],
    max_chunk_tokens: int = 900,
    overlap_tokens: int = 150,
) -> List[Tuple[int, int, str, Optional[str], Dict]]:
    """
    Build variable-sized chunks that respect paragraph boundaries.
    Returns list of tuples: (page_start, page_end, chunk_text, heading, style)
    """
    chunks = []
    cur_texts = []
    cur_tokens = 0
    page_start = None
    page_end = None
    cur_heading = None
    cur_style = {}

    for (pg, para, heading, style) in paragraphs:
        tks = count_tokens(para)
        if cur_tokens + tks <= max_chunk_tokens:
            cur_texts.append(para)
            cur_tokens += tks
            page_end = pg if page_start is not None else pg
            page_start = page_start or pg
            cur_heading = cur_heading or heading
            cur_style = cur_style or (style or {})
        else:
            # emit current chunk
            if cur_texts:
                chunk_text = "\n\n".join(cur_texts).strip()
                chunks.append((page_start or pg, page_end or pg, chunk_text, cur_heading, cur_style))
                # start next with overlap
                if overlap_tokens > 0:
                    # take trailing tail of previous text for continuity
                    tail = truncate_to_last_tokens(chunk_text, overlap_tokens)
                    cur_texts = [tail, para]
                    cur_tokens = count_tokens(tail) + tks
                    page_start = pg  # new start
                    page_end = pg
                    cur_heading = heading
                    cur_style = style or {}
                else:
                    cur_texts = [para]
                    cur_tokens = tks
                    page_start = pg
                    page_end = pg
                    cur_heading = heading
                    cur_style = style or {}
    # final
    if cur_texts:
        chunk_text = "\n\n".join(cur_texts).strip()
        chunks.append((page_start or 1, page_end or (page_start or 1), chunk_text, cur_heading, cur_style))
    return chunks

def truncate_to_last_tokens(s: str, token_budget: int) -> str:
    words = s.split()
    # crude: trim until approx tokens <= budget
    while words and count_tokens(" ".join(words)) > token_budget:
        words = words[1:]
    return " ".join(words).strip()


# ============
# JSONL I/O
# ============

def write_chunks_jsonl(chunks: List[Chunk], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

# ===========================
# Google Cloud: GCS utilities
# ===========================

def upload_to_gcs(local_path: str, bucket_name: str, blob_path: str):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded to gs://{bucket_name}/{blob_path}")

# =========================================
# Vertex AI STUBS (fill these in as needed)
# =========================================

def create_vector_index_stub(
    project: str,
    location: str,
    gcs_embeddings_uri: str,
    index_display_name: str,
):
    """
    STUB: Create a Vertex AI Vector Search (Matching Engine) index from embeddings.
    Steps you will implement:
    - Create embeddings (e.g., text-embedding-004) for each chunk and write to GCS.
    - Use aiplatform.MatchingEngineIndex to create index from the embeddings GCS files.
    - Deploy it to a MatchingEngineIndexEndpoint.
    """
    print("[STUB] create_vector_index: Provide embeddings at", gcs_embeddings_uri)
    print("[STUB] Then use Vertex AI Vector Search to build and deploy the index.")
    print("       See: aiplatform.MatchingEngineIndex and MatchingEngineIndexEndpoint")

def finetune_gemini_stub(
    project: str,
    location: str,
    training_jsonl_gcs_uri: str,
    base_model: str,
    display_name: str,
    epochs: int = 3,
):
    """
    STUB: Fine-tune Gemini with supervised prompt/response JSONL.
    Required JSONL shape typically looks like:
      {"input_text": "<prompt>", "output_text": "<ideal response>"}
    or similar (consult current Vertex docs).
    """
    print("[STUB] finetune_gemini: Supervised tuning requires prompt/response pairs.")
    print(f"       Train file: {training_jsonl_gcs_uri}")
    print(f"       Base model: {base_model}, epochs={epochs}, display_name={display_name}")
    print("       Use vertexai / google-cloud-aiplatform tuning API per current docs.")

def validate_models_stub(
    project: str,
    location: str,
    validation_prompts_path: str,
    baseline_model: str,
    tuned_model_name: Optional[str] = None,
):
    """
    STUB: Run your validation prompts against baseline and tuned model, compare outputs.
    - Use Batch Prediction if supported or loop with online calls.
    - Score with simple heuristics or human-rated labels.
    """
    print("[STUB] validate_models: Compare", baseline_model, "vs", tuned_model_name or "(no tuned model)")
    print("       Prompts file:", validation_prompts_path)

def distill_to_smaller_model_stub(
    teacher_model_name: str,
    training_jsonl_gcs_uri: str,
    student_model_hint: str = "distilbert-base-uncased",
):
    """
    STUB: Knowledge distillation:
    - Generate teacher outputs for prompts (soft targets)
    - Fine-tune a small open-source model on those inputs/outputs
    - Package & serve student for cheap inference (e.g., on Cloud Run or Vertex Endpoint)
    """
    print("[STUB] distill_to_smaller_model: teacher=", teacher_model_name, "student=", student_model_hint)
    print("       Train data:", training_jsonl_gcs_uri)

def deploy_model_stub(
    project: str,
    location: str,
    model_name: str,
    machine_type: str = "n1-standard-2",
):
    """
    STUB: Deploy tuned model to an endpoint (if applicable).
    For Gemini, you'd typically call the model directly; tuned variants can be published.
    """
    print("[STUB] deploy_model:", model_name, "on", machine_type)


# =====================
# Orchestration helpers
# =====================

def build_chunks_from_pdfs(
    pdf_dir: str,
    max_chunk_tokens: int,
    overlap_tokens: int,
    industry: Optional[str] = None,
    risk_factor: Optional[str] = None,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    pdf_paths = sorted([str(p) for p in Path(pdf_dir).glob("*.pdf")])
    if not pdf_paths:
        print(f"[WARN] No PDFs found under: {pdf_dir}")
    for pdf_path in pdf_paths:
        pages = extract_pdf_text_with_layout(pdf_path)
        paragraphs = pages_to_paragraphs(pages)
        para_chunks = chunk_paragraphs(paragraphs, max_chunk_tokens, overlap_tokens)
        for idx, (pstart, pend, text, heading, style) in enumerate(para_chunks, start=1):
            ch = Chunk(
                id=f"{Path(pdf_path).stem}__{idx:05d}",
                text=text,
                source_pdf=os.path.basename(pdf_path),
                page_start=pstart,
                page_end=pend,
                heading=heading,
                style=style,
                industry=industry,
                risk_factor=risk_factor,
            )
            chunks.append(ch)
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="GCP project ID")
    ap.add_argument("--location", default="europe-west4", help="Vertex location")
    ap.add_argument("--gcs_bucket", required=True, help="GCS bucket for outputs")
    ap.add_argument("--output_prefix", required=True, help="GCS prefix, e.g. corpora/industry_risks")
    ap.add_argument("--pdf_dir", required=True, help="Local folder with PDFs")
    ap.add_argument("--max_chunk_tokens", type=int, default=900)
    ap.add_argument("--overlap_tokens", type=int, default=150)
    ap.add_argument("--industry", default=None)
    ap.add_argument("--risk_factor", default=None)
    ap.add_argument("--validation_prompts", default=None, help="Local JSONL of prompts for validation")
    ap.add_argument("--do_embeddings_index", action="store_true", help="Run vector index STUB step")
    ap.add_argument("--do_finetune", action="store_true", help="Run fine-tuning STUB step")
    ap.add_argument("--do_validate", action="store_true", help="Run validation STUB step")
    ap.add_argument("--do_distill", action="store_true", help="Run distillation STUB step")
    ap.add_argument("--do_deploy", action="store_true", help="Run deploy STUB step")
    ap.add_argument("--base_model", default="gemini-2.5-pro", help="Base model name hint")
    ap.add_argument("--tuned_display_name", default="rag-tuned-gemini", help="Display name for tuned model")
    args = ap.parse_args()

    local_jsonl = "chunks.jsonl"

    print("==> Extracting, cleaning, chunking PDFs...")
    chunks = build_chunks_from_pdfs(
        pdf_dir=args.pdf_dir,
        max_chunk_tokens=args.max_chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        industry=args.industry,
        risk_factor=args.risk_factor,
    )
    print(f"   Produced {len(chunks)} chunks")

    print("==> Writing JSONL:", local_jsonl)
    write_chunks_jsonl(chunks, local_jsonl)

    print("==> Uploading to GCS...")
    gcs_chunks_path = f"{args.output_prefix.rstrip('/')}/chunks.jsonl"
    upload_to_gcs(local_jsonl, args.gcs_bucket, gcs_chunks_path)
    gcs_chunks_uri = f"gs://{args.gcs_bucket}/{gcs_chunks_path}"
    print("   Uploaded:", gcs_chunks_uri)

    # === Vector index STUB (for RAG)
    if args.do_embeddings_index:
        print("==> [STUB] Creating embeddings & vector index")
        # Your code: produce embeddings JSONL in GCS (ids + vectors + metadata)
        gcs_embeddings_uri = f"gs://{args.gcs_bucket}/{args.output_prefix.rstrip('/')}/embeddings/embeddings.jsonl"
        create_vector_index_stub(
            project=args.project,
            location=args.location,
            gcs_embeddings_uri=gcs_embeddings_uri,
            index_display_name="rag-index",
        )

    # === Fine-tuning STUB (requires supervised data)
    if args.do_finetune:
        print("==> [STUB] Fine-tuning Gemini (supervised)")
        # You must prepare a *separate* JSONL with prompt/response pairs, e.g.:
        # {"input_text": "...", "output_text": "..."}
        training_jsonl_gcs_uri = f"gs://{args.gcs_bucket}/{args.output_prefix.rstrip('/')}/supervised/train.jsonl"
        finetune_gemini_stub(
            project=args.project,
            location=args.location,
            training_jsonl_gcs_uri=training_jsonl_gcs_uri,
            base_model=args.base_model,
            display_name=args.tuned_display_name,
            epochs=3,
        )

    # === Validation STUB
    if args.do_validate:
        print("==> [STUB] Validation harness")
        baseline_model = args.base_model
        tuned_model_name = args.tuned_display_name  # or the resource name from tuning
        if not args.validation_prompts:
            print("[WARN] --validation_prompts not provided; expected local JSONL with {'prompt': '...'} lines")
        validate_models_stub(
            project=args.project,
            location=args.location,
            validation_prompts_path=args.validation_prompts or "validation_prompts.jsonl",
            baseline_model=baseline_model,
            tuned_model_name=tuned_model_name,
        )

    # === Distillation STUB
    if args.do_distill:
        print("==> [STUB] Distillation")
        training_jsonl_gcs_uri = f"gs://{args.gcs_bucket}/{args.output_prefix.rstrip('/')}/supervised/train.jsonl"
        distill_to_smaller_model_stub(
            teacher_model_name=args.tuned_display_name or args.base_model,
            training_jsonl_gcs_uri=training_jsonl_gcs_uri,
            student_model_hint="distilbert-base-uncased",
        )

    # === Deployment STUB
    if args.do_deploy:
        print("==> [STUB] Deployment")
        deploy_model_stub(
            project=args.project,
            location=args.location,
            model_name=args.tuned_display_name,
            machine_type="n1-standard-2",
        )

    print("Done.")

if __name__ == "__main__":
    main()
