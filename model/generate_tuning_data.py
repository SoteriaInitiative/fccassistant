import os, re, json
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple

from google.cloud import storage
from config import (
    WORKDIR,
    OUT_NAME,
    MAX_PAIRS,
    MIN_CHARS,
    MAX_MODEL_CHARS,
    INSTRUCTION,
    GCS_CORPUS_BUCKET,   # can be "gs://bucket" or "bucket[/prefix]"
    TUNING_PREFIX,       # e.g. "tuning"
)

def _bucket_and_prefix(gs_or_name: str) -> Tuple[str, str]:
    v = gs_or_name.strip()
    if v.startswith("gs://"):
        parts = v[5:].split("/", 1)
        return parts[0], (parts[1] if len(parts) > 1 else "")
    return v, ""

def _gcs_uri_for(filename: str) -> str:
    bucket, base = _bucket_and_prefix(GCS_CORPUS_BUCKET)
    pieces = [p for p in [base, TUNING_PREFIX, filename] if p]
    return f"gs://{bucket}/" + "/".join(pieces)

def _upload_to_gcs(local_path: Path, dest_filename: str) -> str:
    bucket_name, base = _bucket_and_prefix(GCS_CORPUS_BUCKET)
    object_path = "/".join([p for p in [base, TUNING_PREFIX, dest_filename] if p])
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{object_path}"

def load_chunks_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
def naive_summary(text: str, max_chars: int = MAX_MODEL_CHARS) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    sents = _SENT_SPLIT.split(t)
    keep = sents[:2] if len(sents) > 1 else sents[:1]
    if len(" ".join(keep)) < max_chars // 2 and len(sents) >= 3:
        keep = sents[:3]
    return " ".join(keep)[:max_chars].rstrip()

def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if "contents" not in row or not isinstance(row["contents"], list):
        raise ValueError("row missing top-level 'contents' list")
    contents = row["contents"]
    if len(contents) < 2:
        raise ValueError("row must have at least two turns (user, model)")

    user_turn, model_turn = contents[0], contents[1]
    user_turn["role"] = "user"
    model_turn["role"] = "model"  # force 'model' (not 'assistant'/'system')

    for turn in (user_turn, model_turn):
        parts = turn.get("parts")
        if not isinstance(parts, list) or not parts or "text" not in parts[0]:
            raise ValueError("each turn must have non-empty parts with a 'text' field")
        txt = str(parts[0]["text"])
        parts[0]["text"] = txt.replace("\r\n", "\n").strip()

    return {"contents": [user_turn, model_turn]}

def make_row(user_text: str, model_text: str) -> Dict[str, Any]:
    return normalize_row({
        "contents": [
            {"role": "user", "parts": [{"text": user_text}]},
            {"role": "model", "parts": [{"text": model_text}]},
        ]
    })

def main():
    workdir = Path(WORKDIR)
    src = workdir / "corpus_chunks.jsonl"
    dst = workdir / OUT_NAME

    if not src.exists():
        raise SystemExit(f"❌ Missing source: {src}")

    n_out = 0
    dst.parent.mkdir(parents=True, exist_ok=True)

    with dst.open("w", encoding="utf-8") as fout:
        for rec in load_chunks_jsonl(src):
            text = (rec.get("text") or "").strip()
            if len(text) < MIN_CHARS:
                continue
            user_text = f"{INSTRUCTION}\n\n{text}"
            model_text = naive_summary(text)
            fout.write(json.dumps(make_row(user_text, model_text), ensure_ascii=False) + "\n")
            n_out += 1
            if n_out >= MAX_PAIRS:
                break

    print(f"✅ Wrote {n_out} examples to {dst}")

    # Upload to GCS
    uri = _upload_to_gcs(dst, OUT_NAME)
    print(f"☁️  Uploaded to {uri}")
    print("   • Schema: Generate Content JSONL (`contents`, roles user/model)")
    print("   • Ready for Vertex tuning.")

if __name__ == "__main__":
    main()
