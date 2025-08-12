# tools/normalize_gemini_jsonl.py
import io, json, sys

src = sys.argv[1]      # input JSONL in Generate Content schema
dst = sys.argv[2]      # output JSONL

def fix_row(o, i):
    if "contents" not in o or not isinstance(o["contents"], list):
        raise ValueError(f"row {i}: missing contents[]")
    c = o["contents"]
    if len(c) < 2:
        raise ValueError(f"row {i}: contents must have at least 2 turns")
    # keep only first two turns, force roles
    u, m = c[0], c[1]
    u["role"] = "user"
    m["role"] = "model"
    # clean role strings (strip, lowercase enforced above)
    for t in (u, m):
        # ensure parts -> text
        parts = t.get("parts")
        if not isinstance(parts, list) or not parts or "text" not in parts[0]:
            raise ValueError(f"row {i}: each turn needs parts[0].text")
        # strip weird whitespace in role (safety)
        t["role"] = t["role"].strip()
    return {"contents":[u, m]}

with io.open(src, "r", encoding="utf-8-sig") as fin, io.open(dst, "w", encoding="utf-8") as fout:
    n = 0
    for i, line in enumerate(fin, 1):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        fixed = fix_row(o, i)
        fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")
        n += 1
print(f"✅ wrote {n} lines -> {dst}")
