# app/main.py
import os, json, time, secrets, re
from typing import Dict, Any, Optional, List

import numpy as np
import faiss
import logging

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Vertex / ADK
from vertexai import init as vertexai_init
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.reasoning_engines import AdkApp
from google.adk.agents import Agent
import boto3

from google.cloud import storage
_storage_client: Optional[storage.Client] = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fcc-ai-agent")

# ------------------------------
# Env configuration (provider-agnostic)
# ------------------------------
PROJECT_ID         = os.getenv("PROJECT_ID")
LOCATION           = os.getenv("LOCATION", "us-central1")
BASE_MODEL_NAME    = os.getenv("BASE_MODEL_NAME", "")  # publisher path or short; prefer full publisher path
TUNED_MODEL_NAME   = os.getenv("TUNED_MODEL_NAME", "") # endpoint path when tuned, e.g.: projects/.../endpoints/...
ROUTER_MODEL_NAME  = os.getenv(
    "ROUTER_MODEL_NAME",
    (f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-2.0-flash"
     if PROJECT_ID else "publishers/google/models/gemini-2.0-flash")
)

def _parse_gcs_prefix(s: Optional[str]) -> tuple[str, str]:
    """Parse env GCS_PREFIX into (bucket, prefix) safely.
    Accepts forms: "", "bucket", "bucket/prefix", "gs://bucket", "gs://bucket/prefix".
    Returns ("", "") if empty/invalid.
    """
    s = (s or "").strip()
    if not s:
        return "", ""
    if s.startswith("gs://"):
        s = s[5:]
    s = s.strip("/")
    if not s:
        return "", ""
    parts = s.split("/", 1)
    bucket = parts[0].strip()
    prefix = parts[1].strip("/") if len(parts) > 1 else ""
    if not bucket:
        return "", ""
    return bucket, prefix

GCS_BUCKET, GCS_PREFIX = _parse_gcs_prefix(os.getenv("GCS_PREFIX", ""))
WORKDIR            = os.getenv("WORKDIR", "/workspace")  # folder that has embeddings/faiss/corpus
ALLOW_CORS_ALL     = os.getenv("ALLOW_CORS_ALL", "1").lower() in ("1", "true", "yes")
ALLOWED_USERS      = os.getenv("ALLOWED_USERS", "")      # "email:pw,email2:pw2"

# AWS Bedrock (optional) — if enabled, the app uses Bedrock KB RAG instead of Vertex + local FAISS
AWS_BEDROCK_MODE   = os.getenv("AWS_BEDROCK_MODE", "0").lower() in ("1", "true", "yes")
AWS_REGION         = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
BEDROCK_KB_ID      = os.getenv("BEDROCK_KB_ID", "")
BEDROCK_MODEL_ARN  = os.getenv("BEDROCK_MODEL_ARN", "")  # use PT ARN if available
BEDROCK_BASE_MODEL_ID = os.getenv("BEDROCK_BASE_MODEL_ID", "amazon.nova-micro-v1:0")

DECLINE_MESSAGE    = os.getenv("DECLINE_MESSAGE", "I can’t process non‑sanctions questions at the moment. This assistant is restricted to sanctions-related queries.")

# Optional floor for sanctions top_k (set to 0 to disable)
SANCTIONS_TOPK_FLOOR = int(os.getenv("SANCTIONS_TOPK_FLOOR", "5"))

GCS_CACHE_DIR      = os.getenv("GCS_CACHE_DIR", "/tmp/fcc_cache")
os.makedirs(GCS_CACHE_DIR, exist_ok=True)

# ------------------------------
# FastAPI app + CORS
# ------------------------------
app = FastAPI(title="FCC AI Assistant")

if ALLOW_CORS_ALL:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

# ------------------------------
# Simple env-based auth
# ------------------------------
TOKENS: Dict[str, str] = {}  # token -> email

def _get_storage() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client(project=PROJECT_ID) if PROJECT_ID else storage.Client()
    return _storage_client

def _gcs_key(name: str) -> str:
    # Compose "prefix/name" if prefix present
    return f"{GCS_PREFIX}/{name}" if GCS_PREFIX else name

def _ensure_local_from_gcs(name: str) -> str:
    """
    Ensure the given artifact (embeddings.npy, faiss.index, embeddings_meta.jsonl, corpus_chunks.jsonl)
    exists locally under GCS_CACHE_DIR. If missing, download from gs://GCS_BUCKET/prefix/name.
    Returns local path.
    """
    local_path = os.path.join(GCS_CACHE_DIR, name)
    if os.path.exists(local_path):
        return local_path
    if not GCS_BUCKET:
        # No bucket configured -> fallback to WORKDIR
        fallback = os.path.join(WORKDIR, name)
        if not os.path.exists(fallback):
            raise FileNotFoundError(f"No GCS bucket and local file missing: {fallback}")
        return fallback

    bucket = _get_storage().bucket(GCS_BUCKET)
    blob = bucket.blob(_gcs_key(name))
    if not blob.exists():
        raise FileNotFoundError(f"Blob not found in GCS: gs://{GCS_BUCKET}/{_gcs_key(name)}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path

def _allowed_map() -> Dict[str, str]:
    m = {}
    s = ALLOWED_USERS.strip()
    if not s:
        return m
    for pair in s.split(","):
        if ":" in pair:
            email, pw = pair.split(":", 1)
            m[email.strip().lower()] = pw
    return m

def _auth_email_from_token(token: Optional[str]) -> Optional[str]:
    if not token: return None
    return TOKENS.get(token)

def _require_auth(request: Request) -> str:
    # Bearer
    auth = request.headers.get("Authorization", "")
    email = None
    if auth.lower().startswith("bearer "):
        email = _auth_email_from_token(auth.split(" ", 1)[1].strip())
    # or ?token= (for EventSource/SSE)
    if not email:
        token = request.query_params.get("token")
        if token:
            email = _auth_email_from_token(token)
    # dev mode: if no users defined
    if not email and not _allowed_map():
        email = "dev@local"
    if not email:
        raise HTTPException(401, "Unauthorized")
    return email

# ------------------------------
# RAG tool: FAISS retrieval
# ------------------------------
def load_index():
    # ✨ now resolves from GCS (cached locally), or falls back to WORKDIR if no GCS configured
    emb_path  = _ensure_local_from_gcs("embeddings.npy")
    faiss_path = _ensure_local_from_gcs("faiss.index")
    meta_path  = _ensure_local_from_gcs("embeddings_meta.jsonl")

    vecs = np.load(emb_path)
    index = faiss.read_index(faiss_path)

    meta: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return vecs, index, meta

def embed_query(q: str):
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return model.get_embeddings([q])[0].values

def retrieve(q: str, k: int = 5) -> List[str]:
    _, index, meta = load_index()
    qv = np.array(embed_query(q), dtype="float32")[None, :]
    faiss.normalize_L2(qv)
    _, idxs = index.search(qv, k)
    # Load raw chunks
    raw_path = _ensure_local_from_gcs("corpus_chunks.jsonl")
    raw: Dict[str, str] = {}
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            raw[f'{r["doc"]}:{r["chunk_id"]}'] = r["text"]
    contexts: List[str] = []
    for i in idxs[0]:
        h = meta[i]
        key = f'{h["doc"]}:{h["chunk_id"]}'
        contexts.append(raw.get(key, ""))
    return contexts

def search_index(query: str, top_k: int = 5) -> list[str]:
    """ADK tool: Return up to top_k context chunks from local FAISS index."""
    top_k = max(1, min(int(top_k or 5), 15))
    return retrieve(query, k=top_k)

# ------------------------------
# AWS Bedrock KB RAG helpers
# ------------------------------
def _bedrock_model_arn() -> str:
    if BEDROCK_MODEL_ARN:
        return BEDROCK_MODEL_ARN
    return f"arn:aws:bedrock:{AWS_REGION}::foundation-model/{BEDROCK_BASE_MODEL_ID}"

def bedrock_retrieve_and_generate(question: str, top_k: int = 5) -> str:
    """Call Bedrock KB retrieve-and-generate using KB ID and model ARN from env."""
    if not BEDROCK_KB_ID:
        raise HTTPException(500, "BEDROCK_KB_ID not configured for AWS mode")
    client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
    cfg = {
        "vectorSearchConfiguration": {
            "numberOfResults": max(1, min(int(top_k or 5), 15))
        }
    }
    req = {
        "knowledgeBaseId": BEDROCK_KB_ID,
        "modelArn": _bedrock_model_arn(),
        "retrievalQuery": {"text": question},
        "retrievalConfiguration": cfg,
    }
    resp = client.retrieve_and_generate(**req)
    out = resp.get("output", {}).get("text") or ""
    return out.strip()

# ------------------------------
# Agents
# ------------------------------
SANCTIONS_INSTRUCTION = (
    "You are an experienced US Sanctions Officer. "
    "Use ONLY the context returned by the search_index tool. "
    "If the answer is not fully supported by context, say you don't know."
)

GENERAL_INSTRUCTION = (
    "You do not answer the question. Kindly decline to analyse the user question and explain you are not built for general use."
)

DOMAIN_PROFILES = {
    "sanctions": {
        "model": lambda use_tuned: (TUNED_MODEL_NAME if use_tuned and TUNED_MODEL_NAME else BASE_MODEL_NAME),
        "instruction": SANCTIONS_INSTRUCTION,
        "use_tool": True,
    },
    "general": {
        "model": lambda _use: os.getenv("GENERAL_MODEL_NAME", ROUTER_MODEL_NAME),  # harmless default
        "instruction": GENERAL_INSTRUCTION,
        "use_tool": False,
    },
}

def build_router_agent() -> Agent:
    return Agent(
        model=ROUTER_MODEL_NAME,
        name="router",
        instruction=(
            "You are a strict router. Output ONLY JSON.\n"
            "If the question mentions sanctions concepts (OFAC, SDN, 50% rule, blocking, FAQ 401, EU/UK/SECO), "
            "set domain='sanctions'; else 'general'.\n"
            "Rules:\n"
            " - For domain='general', set top_k=0 and use_tuned=false.\n"
            " - For domain='sanctions', choose a reasonable top_k (5..15).\n"
            "Return JSON: {domain:'sanctions'|'general', top_k:5..15, use_tuned:bool, task:str}\n"
            "Never include prose—JSON only."
        ),
    )

def build_reasoner_agent_for(domain: str, use_tuned: bool) -> Agent:
    prof = DOMAIN_PROFILES.get(domain, DOMAIN_PROFILES["general"])
    model_name = prof["model"](use_tuned)
    tools = [search_index] if prof["use_tool"] else []
    return Agent(
        model=model_name,
        name=f"reasoner_{domain}",
        tools=tools,
        instruction=prof["instruction"],
    )

# ------------------------------
# Schemas
# ------------------------------
class ChatIn(BaseModel):
    query: str
    top_k: Optional[int] = 5
    model: Optional[str] = "base"   # "base" | "tuned" (hint; router may override)

# ------------------------------
# Health & Auth
# ------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/api/login")
def login(payload: Dict[str, str]):
    email = (payload.get("email") or "").strip().lower()
    pw = payload.get("password") or ""
    allowed = _allowed_map()
    if allowed:
        if email not in allowed or allowed[email] != pw:
            raise HTTPException(401, "Invalid credentials")
    token = secrets.token_urlsafe(24)
    TOKENS[token] = email or "dev@local"
    return {"token": token}

@app.post("/api/logout")
def logout(request: Request):
    token = None
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    token = token or request.query_params.get("token")
    if token and token in TOKENS:
        TOKENS.pop(token, None)
    return {"ok": True}

# ------------------------------
# Non-streaming final JSON endpoint (back-compat)
# ------------------------------
@app.post("/api/chat")
def chat(payload: ChatIn, request: Request, email: str = Depends(_require_auth)):
    if AWS_BEDROCK_MODE:
        # Direct Bedrock KB RAG path
        try:
            ans = bedrock_retrieve_and_generate(payload.query, payload.top_k or 5)
        except Exception as e:
            raise HTTPException(500, f"Bedrock error: {e}")
        return {"answer": ans, "contexts": []}

    vertexai_init(project=PROJECT_ID, location=LOCATION)

    # 1) Router (non-streaming)
    router_app = AdkApp(agent=build_router_agent())
    router_text = ""
    for ev in router_app.stream_query(user_id=email, message=f"User question:\n{payload.query}\n\nJSON only:"):
        parts = (ev.get("content") or {}).get("parts", [])
        for p in parts:
            if "text" in p: router_text += p["text"]

    # Parse robustly (strip code fences if present)
    try:
        rt = router_text.strip()
        if rt.startswith("```"):
            rt = "\n".join(line for line in rt.splitlines() if not line.strip().startswith("```"))
        if not (rt.startswith("{") and rt.endswith("}")):
            m = re.search(r"\{[\s\S]*\}", rt)
            if m: rt = m.group(0)
        plan = json.loads(rt or "{}")
    except Exception:
        plan = {}

    domain = (plan.get("domain") or "general").strip().lower()
    try:
        k = int(plan.get("top_k", payload.top_k or 10))
    except Exception:
        k = payload.top_k or 10
    k = max(0, min(k, 10))
    if domain == "sanctions" and SANCTIONS_TOPK_FLOOR > 0 and k < SANCTIONS_TOPK_FLOOR:
        k = SANCTIONS_TOPK_FLOOR
    if domain != "sanctions":
        # General path: do not reason; just return decline
        return {"answer": DECLINE_MESSAGE, "contexts": []}

    use_tuned = bool(plan.get("use_tuned", payload.model == "tuned"))
    task = (plan.get("task") or payload.query).strip()

    # 2) Reasoner (non-streaming)
    agent = build_reasoner_agent_for(domain, use_tuned)
    app_agent = AdkApp(agent=agent)
    if DOMAIN_PROFILES["sanctions"]["use_tool"] and k > 0:
        prompt = (
            f"Use search_index(query='{payload.query}', top_k={k}) to fetch context.\n"
            f"Then answer the task: {task}\n"
            f"Respond strictly based on retrieved context. If unsupported, say you don't know."
        )
    else:
        prompt = f"Task: {task}\nRespond strictly based on retrieved context. If unsupported, say you don't know."

    final_answer = ""
    for ev in app_agent.stream_query(user_id=email, message=prompt):
        parts = (ev.get("content") or {}).get("parts", [])
        for p in parts:
            if "text" in p: final_answer += p["text"]

    return {"answer": (final_answer or "").strip(), "contexts": []}

# ------------------------------
# Streaming SSE endpoint (/api/chat/stream)
# ------------------------------
def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

@app.get("/api/chat/stream")
async def chat_stream(request: Request, q: str, top_k: int = 5, model: str = "base", token: str = "", rid: str = ""):
    email = _require_auth(request)
    if AWS_BEDROCK_MODE:
        # For AWS mode, do a one-shot Bedrock call and stream minimal events
        async def gen_bedrock():
            yield sse("route", {"domain": "sanctions", "top_k": max(1, top_k), "use_tuned": (BEDROCK_MODEL_ARN != ""), "task": q, "model": _bedrock_model_arn()})
            try:
                ans = bedrock_retrieve_and_generate(q, top_k)
            except Exception as e:
                ans = f"Bedrock error: {e}"
            yield sse("final", {"answer": ans})
        headers = {
            "Cache-Control": "no-cache",
            "Content-Type": "text/event-stream",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(gen_bedrock(), headers=headers)

    vertexai_init(project=PROJECT_ID, location=LOCATION)

    async def gen():
        # 1) Router
        router_app = AdkApp(agent=build_router_agent())
        router_text = ""
        for ev in router_app.stream_query(user_id=email, message=f"User question:\n{q}\n\nJSON only:"):
            parts = (ev.get("content") or {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    router_text += p["text"]

        # Robust parse
        plan = {}
        try:
            rt = router_text.strip()
            if rt.startswith("```"):
                rt = "\n".join(line for line in rt.splitlines() if not line.strip().startswith("```"))
            if not (rt.startswith("{") and rt.endswith("}")):
                m = re.search(r"\{[\s\S]*\}", rt)
                if m: rt = m.group(0)
            plan = json.loads(rt or "{}")
        except Exception:
            plan = {}

        domain = (plan.get("domain") or "general").strip().lower()
        try:
            k = int(plan.get("top_k", top_k or 10))
        except Exception:
            k = top_k or 5
        k = max(0, min(k, 10))
        if domain == "sanctions" and SANCTIONS_TOPK_FLOOR > 0 and k < SANCTIONS_TOPK_FLOOR:
            k = SANCTIONS_TOPK_FLOOR

        if domain != "sanctions":
            # general → do not reason; return a final decline
            yield sse("route", {"domain": domain, "top_k": 0, "use_tuned": False, "task": plan.get("task", q), "model": None})
            yield sse("final", {"answer": DECLINE_MESSAGE})
            return

        use_tuned = bool(plan.get("use_tuned", model == "tuned"))
        task = (plan.get("task") or q).strip()

        agent = build_reasoner_agent_for(domain, use_tuned)
        yield sse("route", {"domain": domain, "top_k": k, "use_tuned": use_tuned, "task": task, "model": agent.model})

        # 2) Reasoner
        app_agent = AdkApp(agent=agent)
        if DOMAIN_PROFILES["sanctions"]["use_tool"] and k > 0:
            prompt = (
                f"Use search_index(query='{q}', top_k={k}) to fetch context.\n"
                f"Then answer the task: {task}\n"
                f"Respond strictly based on retrieved context. If unsupported, say you don't know."
            )
        else:
            prompt = f"Task: {task}\nRespond strictly based on retrieved context. If unsupported, say you don't know."

        answer_buf: List[str] = []
        for ev in app_agent.stream_query(user_id=email, message=prompt):
            parts = (ev.get("content") or {}).get("parts", [])
            for p in parts:
                if "function_call" in p:
                    yield sse("retrieve", {
                        "tool": p["function_call"].get("name", "search_index"),
                        "args": p["function_call"].get("args", {})
                    })
                if "text" in p:
                    chunk = p["text"]
                    answer_buf.append(chunk)
                    yield sse("delta", {"text": chunk})
        yield sse("final", {"answer": "".join(answer_buf).strip()})

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), headers=headers)

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
