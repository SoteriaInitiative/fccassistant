# tools/count_tokens_jsonl.py
import argparse, csv, json, math, sys
from pathlib import Path

# --- optional backends ---
def _heuristic_tokens(s: str) -> int:
    words = len(s.split())
    chars = len(s)
    return max(math.ceil(words * 1.33), math.ceil(chars / 4))

def _tiktoken_tokens(s: str):
    try:
        import tiktoken
    except Exception as e:
        raise RuntimeError("tiktoken not installed. pip install tiktoken") from e
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(s))

def _vertex_count_tokens(texts, model_name, project, location):
    # Uses Vertex AI CountTokens API via the GenerativeModel (works well as an upper bound).
    # Requires: pip install google-cloud-aiplatform google-generativeai
    from vertexai import init as vertexai_init
    from vertexai.generative_models import GenerativeModel
    vertexai_init(project=project, location=location)
    model = GenerativeModel(model_name)
    out = []
    for t in texts:
        resp = model.count_tokens(t)
        out.append(int(resp.total_tokens))
    return out
# --------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="corpus_chunks.jsonl", help="Path to corpus jsonl")
    p.add_argument("--out", default="token_counts.csv", help="CSV output path")
    p.add_argument("--mode", choices=["heuristic","tiktoken","vertex"], default="heuristic")
    p.add_argument("--vertex_model", default="gemini-1.5-pro-002",
                   help="Only if --mode=vertex: model for CountTokens")
    p.add_argument("--project", help="Only if --mode=vertex")
    p.add_argument("--location", default="us-central1", help="Only if --mode=vertex")
    p.add_argument("--limit", type=int, default=20000, help="Token limit to flag")
    p.add_argument("--top", type=int, default=10, help="Show top-N largest chunks in stdout")
    args = p.parse_args()

    src = Path(args.jsonl)
    if not src.exists():
        sys.exit(f"❌ File not found: {src}")

    rows = []
    texts = []
    recs = []

    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            recs.append(rec)
            texts.append(text)

    if args.mode == "vertex":
        if not args.project:
            sys.exit("❌ --project is required for --mode=vertex")
        tokens_list = _vertex_count_tokens(texts, args.vertex_model, args.project, args.location)
    elif args.mode == "tiktoken":
        tokens_list = [_tiktoken_tokens(t) for t in texts]
    else:
        tokens_list = [_heuristic_tokens(t) for t in texts]

    for rec, toks in zip(recs, tokens_list):
        text = rec.get("text", "")
        rows.append({
            "doc": rec.get("doc"),
            "chunk_id": rec.get("chunk_id"),
            "chars": len(text),
            "words": len(text.split()),
            "tokens": toks,
            "over_limit": int(toks > args.limit),
        })

    # Write CSV
    outp = Path(args.out)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc","chunk_id","chars","words","tokens","over_limit"])
        w.writeheader()
        w.writerows(rows)

    # Show a quick summary
    total = len(rows)
    overs = sum(r["over_limit"] for r in rows)
    print(f"✅ Analyzed {total} chunks. Over {args.limit} tokens: {overs}")
    print(f"CSV written to: {outp}")

    # Top-N largest
    top = sorted(rows, key=lambda r: r["tokens"], reverse=True)[:args.top]
    print("\nTop offenders:")
    for r in top:
        print(f"- {r['doc']}  chunk={r['chunk_id']}  tokens≈{r['tokens']}  chars={r['chars']}  over={r['over_limit']}")

if __name__ == "__main__":
    main()
