import os
import json
import time
import threading
import secrets
import uuid
import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from fastapi import FastAPI, Depends, HTTPException, Header, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.cloud import storage
from vertexai import init as vertexai_init
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ask-simons")

def _timer(name: str):
    t0 = time.time()
    def done(**extra):
        dt_ms = round((time.time() - t0) * 1000, 1)
        log.info("timing.%s", name, extra={"ms": dt_ms, **extra})
    return done

# ---------- Config ----------
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "gemini-2.5-flash")
TUNED_MODEL_NAME = os.getenv("TUNED_MODEL_NAME", "")
GCS_PREFIX = os.getenv("GCS_PREFIX")
ALLOW_CORS_ALL = os.getenv("ALLOW_CORS_ALL", "1") == "1"

# ---------- Vertex init ----------
if PROJECT_ID:
    vertexai_init(project=PROJECT_ID, location=LOCATION)

# Reuse models across requests (reduce per-call overhead)
_EMB_MODEL = TextEmbeddingModel.from_pretrained("text-embedding-004")
_GEN_BASE = GenerativeModel(BASE_MODEL_NAME)
_GEN_TUNED = GenerativeModel(TUNED_MODEL_NAME) if TUNED_MODEL_NAME else None

# ---------- Storage + index cache ----------
_storage_client = None
_index_lock = threading.Lock()
_cached = {"index": None, "vecs": None, "meta": None, "raw": None, "loaded_at": 0.0}

def _get_storage() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

def _parse_gs(uri: str) -> Tuple[str, str]:
    if not uri or not uri.startswith("gs://"):
        raise ValueError("GCS_PREFIX must be a gs:// URI")
    path = uri[5:]
    bucket, *rest = path.split("/", 1)
    prefix = rest[0] if rest else ""
    return bucket, prefix.rstrip("/")

def _download_bytes(bucket: str, path: str) -> bytes:
    client = _get_storage()
    blob = client.bucket(bucket).blob(path)
    return blob.download_as_bytes()

def _ensure_index_loaded():
    with _index_lock:
        if _cached["index"] is not None and (time.time() - _cached["loaded_at"] < 3600):
            return
        if not GCS_PREFIX:
            raise RuntimeError("GCS_PREFIX env var must be set (gs://bucket/prefix)")
        t = _timer("load_index")
        bucket, prefix = _parse_gs(GCS_PREFIX)
        paths = {
            "emb": f"{prefix}/embeddings.npy",
            "faiss": f"{prefix}/faiss.index",
            "meta": f"{prefix}/embeddings_meta.jsonl",
            "corpus": f"{prefix}/corpus_chunks.jsonl",
        }
        emb_bytes = _download_bytes(bucket, paths["emb"])
        idx_bytes = _download_bytes(bucket, paths["faiss"])
        meta_bytes = _download_bytes(bucket, paths["meta"])
        corpus_bytes = _download_bytes(bucket, paths["corpus"])

        import io, tempfile
        vecs = np.load(io.BytesIO(emb_bytes))
        with tempfile.NamedTemporaryFile(suffix=".index") as tf:
            tf.write(idx_bytes)
            tf.flush()
            index = faiss.read_index(tf.name)
        faiss.normalize_L2(vecs)

        meta = [json.loads(l) for l in meta_bytes.decode("utf-8").splitlines() if l.strip()]
        raw = {}
        for line in corpus_bytes.decode("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                key = f'{r.get("doc")}:{r.get("chunk_id")}'
                raw[key] = r.get("text", "")

        _cached.update({"index": index, "vecs": vecs, "meta": meta, "raw": raw, "loaded_at": time.time()})
        t(ntotal=int(index.ntotal))

# ---------- Models ----------
class ChatRequest(BaseModel):
    query: str
    top_k: int | None = 5
    model: str | None = "base"

class ChatResponse(BaseModel):
    answer: str
    model_used: str
    contexts: List[Dict[str, Any]]

class LoginReq(BaseModel):
    email: str
    password: str

class LoginResp(BaseModel):
    token: str

# ---------- Simple allowlist + sessions ----------
def _load_allowed_users() -> Dict[str, str]:
    raw = os.getenv("ALLOWED_USERS", "").strip()
    users: Dict[str, str] = {}
    if raw:
        for pair in raw.split(","):
            if ":" in pair:
                email, pw = pair.split(":", 1)
                users[email.strip().lower()] = pw.strip()
    return users

ALLOWED = _load_allowed_users()
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "7200"))  # 2h

def _issue_token(email: str) -> str:
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {"email": email, "exp": time.time() + SESSION_TTL_SECONDS}
    return token

def _validate_token(authorization: Optional[str]) -> Dict[str, Any]:
    if os.getenv("DISABLE_AUTH") == "1":
        return {"email": "dev-user", "token": "dev"}
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    sess = SESSIONS.get(token)
    if not sess:
        raise HTTPException(status_code=401, detail="Invalid session")
    if time.time() > float(sess["exp"]):
        SESSIONS.pop(token, None)
        raise HTTPException(status_code=401, detail="Session expired")
    return {"email": sess["email"], "token": token}

def verify_auth(authorization: Optional[str] = Header(None)):
    return _validate_token(authorization)

# ---------- RAG helpers with timing ----------
def embed_query(text: str):
    t = _timer("embed_query")
    try:
        return _EMB_MODEL.get_embeddings([text])[0].values
    finally:
        t(chars=len(text))

def retrieve(query: str, k: int = 5):
    t = _timer("retrieve")
    try:
        _ensure_index_loaded()
        index = _cached["index"]
        meta = _cached["meta"]
        qv = np.array(embed_query(query), dtype="float32")[None, :]
        faiss.normalize_L2(qv)
        sims, idxs = index.search(qv, min(k, index.ntotal))
        hits = []
        for i, score in zip(idxs[0], sims[0]):
            if i == -1:
                continue
            m = meta[i].copy()
            m["_score"] = float(score)
            hits.append(m)
        return hits
    finally:
        ntotal = int(_cached["index"].ntotal) if _cached["index"] is not None else 0
        t(k=k, index_ntotal=ntotal)


def build_context(hits):
    raw = _cached["raw"]
    ctxs = []
    for h in hits:
        key = f'{h.get("doc")}:{h.get("chunk_id")}'
        ctxs.append({
            "doc": h.get("doc"),
            "chunk_id": h.get("chunk_id"),
            "text": raw.get(key, ""),
            "score": h.get("_score", 0.0),
        })
    return ctxs


def generate_answer(query: str, contexts, use_tuned: bool):
    t = _timer("generate_answer")
    model_name = TUNED_MODEL_NAME if (use_tuned and TUNED_MODEL_NAME) else BASE_MODEL_NAME
    model = _GEN_TUNED if (use_tuned and _GEN_TUNED) else _GEN_BASE
    ctx_text = "\n---\n".join(c["text"] for c in contexts if c.get("text"))
    prompt = (
        "You are an experienced US Sanctions Officer. Answer the question strictly using the provided context. "
        "If the answer is not fully supported by context, say you don't know.\n\n"
        f"Context:\n{ctx_text}\n\nQuestion: {query}"
    )
    try:
        log.info("vertex.request", extra={"model": model_name, "prompt_chars": len(prompt), "ctx_chars": len(ctx_text), "q_chars": len(query)})
        resp = model.generate_content(prompt)
        answer = getattr(resp, "text", "").strip() or "I'm not sure."
        return answer, model_name
    finally:
        t(model=model_name)

# ---------- FastAPI app ----------
app = FastAPI(title="Ask Simons - FCC Assistant MVP")

# CORS for non-same-origin testing
if ALLOW_CORS_ALL:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Request logging middleware (request IDs, timings)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = request.headers.get("X-Client-Request-ID") or str(uuid.uuid4())
    t0 = time.time()
    path = request.url.path
    method = request.method
    try:
        log.info("api.request", extra={"rid": rid, "method": method, "path": path})
        response: Response = await call_next(request)
        dt = round((time.time() - t0) * 1000, 1)
        response.headers["X-Request-ID"] = rid
        log.info("api.response", extra={"rid": rid, "method": method, "path": path, "status": response.status_code, "ms": dt})
        return response
    except Exception as e:
        dt = round((time.time() - t0) * 1000, 1)
        log.exception("api.error", extra={"rid": rid, "method": method, "path": path, "ms": dt})
        # Preserve JSON error shape for clients
        return Response(
            content=json.dumps({"detail": f"Internal error: {str(e)}", "request_id": rid}),
            status_code=500,
            media_type="application/json",
            headers={"X-Request-ID": rid},
        )

# ---------- Startup warm-up (so first user is faster) ----------
@app.on_event("startup")
def _warm():
    log.info("startup.begin")
    try:
        _ensure_index_loaded()
        # Touch models once
        _ = _EMB_MODEL.get_embeddings(["warmup"])[0].values
        _ = _GEN_BASE.generate_content("warmup")
        if _GEN_TUNED:
            _ = _GEN_TUNED.generate_content("warmup")
        log.info("startup.done")
    except Exception as e:
        log.exception("startup.error")
        # Don’t crash; health endpoint will reflect readiness issues

# ---------- Health ----------
@app.get("/api/health")
def health():
    try:
        _ensure_index_loaded()
        return {"ok": True, "index_size": int(_cached["index"].ntotal)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- Auth endpoints ----------
@app.post("/api/login", response_model=LoginResp)
def login(req: LoginReq):
    email = (req.email or "").strip().lower()
    pw = (req.password or "")
    if not email or not pw:
        raise HTTPException(status_code=400, detail="Email and password required")
    allowed_pw = _load_allowed_users().get(email)  # reload to pick up new env (optional)
    if not allowed_pw or allowed_pw != pw:
        raise HTTPException(status_code=401, detail="Invalid credentials or not authorized")
    token = _issue_token(email)
    log.info("auth.login", extra={"email": email})
    return LoginResp(token=token)

@app.post("/api/logout")
def logout(user=Depends(verify_auth)):
    SESSIONS.pop(user["token"], None)
    log.info("auth.logout", extra={"email": user["email"]})
    return {"ok": True}

# ---------- Chat ----------
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user=Depends(verify_auth)):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    top_k = max(1, min(int(req.top_k or 5), 20))
    log.info("chat.begin", extra={"email": user.get("email"), "top_k": top_k, "model": req.model, "q_chars": len(req.query)})
    hits = retrieve(req.query, top_k)
    ctxs = build_context(hits)
    ans, used = generate_answer(req.query, ctxs, use_tuned=(req.model == "tuned"))
    log.info("chat.end", extra={"email": user.get("email"), "model_used": used, "ctxs": len(ctxs), "ans_chars": len(ans)})
    return ChatResponse(answer=ans, model_used=used, contexts=ctxs)

# ---------- Static frontend ----------
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
