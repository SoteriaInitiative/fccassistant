import os, json
import numpy as np
import faiss
from typing import List, Dict, Any

from google.cloud import aiplatform
from vertexai import init as vertexai_init
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel
from config import PROJECT_ID, LOCATION, TUNING_BASE_MODEL, WORKDIR, TUNING_DISPLAY_NAME
from google.genai import types
from google.adk.agents import Agent
from vertexai.preview.reasoning_engines import AdkApp



ROUTER_MODEL_NAME = os.getenv("ROUTER_MODEL_NAME", "gemini-2.0-flash")
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", TUNING_BASE_MODEL)
aiplatform.init(project=PROJECT_ID, location=LOCATION)
endpoints = aiplatform.Endpoint.list(
    filter=f'display_name="{TUNING_DISPLAY_NAME}"',
    order_by="update_time desc"
)
TUNED_MODEL_NAME = os.getenv("TUNED_MODEL_NAME",
                             f"projects/{PROJECT_ID}/locations/us-central1/endpoints/{endpoints[0].name}")



def search_index(query: str, top_k: int = 5) -> list[str]:
    """Return up to top_k context chunks from the local FAISS index.
    Args:
      query: natural-language question.
      top_k: number of context chunks (1..10).
    Returns:
      List of plaintext context strings.
    """
    top_k = max(1, min(int(top_k or 5), 10))
    return retrieve(query, k=top_k)



def load_index():
    vecs = np.load(os.path.join(WORKDIR, "embeddings.npy"))
    index = faiss.read_index(os.path.join(WORKDIR, "faiss.index"))
    meta = []
    with open(os.path.join(WORKDIR, "embeddings_meta.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return vecs, index, meta

def embed_query(q: str):

    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return model.get_embeddings([q])[0].values

def retrieve(q: str, k=5):
    vecs, index, meta = load_index()
    qv = np.array(embed_query(q), dtype="float32")[None, :]
    faiss.normalize_L2(qv)
    sims, idxs = index.search(qv, k)
    hits = []
    for i in idxs[0]:
        hits.append(meta[i])
    # Load raw chunks for context
    raw = {}
    with open(os.path.join(WORKDIR, "corpus_chunks.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = f'{r["doc"]}:{r["chunk_id"]}'
            raw[key] = r["text"]
    contexts = []
    for h in hits:
        key = f'{h["doc"]}:{h["chunk_id"]}'
        contexts.append(raw.get(key, ""))
    return contexts

def answer(q: str):
    print(run_with_adk(q, user_id="cli-user"))

# ----------------------------
# Build ADK Agents (Router & Reasoner)
# ----------------------------
def build_router_agent() -> Agent:
    return Agent(
        model=ROUTER_MODEL_NAME,
        name="router",
        instruction=(
            "You are a router for sanctions/compliance Q&A.\n"
            "Return ONLY strict JSON with keys: top_k (1..10), use_tuned (bool), task (str)."
        ),
    )

def build_reasoner_agent() -> Agent:
    model_name = TUNED_MODEL_NAME
    return Agent(
        model=model_name,
        name="reasoner",
        tools=[search_index],
        instruction=(
            "You are an experienced US Sanctions Officer. "
            "Use ONLY the context returned by the search_index tool. "
            "If the answer is not fully supported by context, say you don't know."
        ),
    )

# ----------------------------
# Orchestration with AdkApp + stream_query
# ----------------------------
def run_with_adk(question: str, user_id: str = "local-user") -> str:
    vertexai_init(project=PROJECT_ID, location=LOCATION)

    # Create apps
    router_app = AdkApp(agent=build_router_agent())
    reasoner_agent = build_reasoner_agent()
    reasoner_app = AdkApp(agent=reasoner_agent)

    trace: List[Dict[str, Any]] = []

    # ---- 1) Route: get plan JSON from router
    router_text = ""
    for ev in router_app.stream_query(user_id=user_id, message=f"User question:\n{question}\n\nJSON only:"):
        # You can surface progress here; we collect for a trace.
        if ev.get("content") and ev["content"].get("parts"):
            # concatenate any text parts
            router_text += "".join(p.get("text", "") for p in ev["content"]["parts"])

    try:
        plan = json.loads(router_text.strip() or "{}")
    except Exception:
        plan = {}

    top_k = max(1, min(int(plan.get("top_k", 5)), 10))
    use_tuned = bool(plan.get("use_tuned", True))
    task = (plan.get("task") or question).strip()

    trace.append({"step": "route", "data": {"top_k": top_k, "use_tuned": use_tuned, "task": task}})

    # If router requests tuned and TUNED_MODEL_NAME is set, rebuild reasoner on tuned
    tuned_name = os.getenv("TUNED_MODEL_NAME")
    if use_tuned and tuned_name and tuned_name != reasoner_agent.model:
        reasoner_agent = Agent(
            model=tuned_name,
            name="reasoner",
            tools=[search_index],
            system_instruction=reasoner_agent.system_instruction,
        )
        reasoner_app = AdkApp(agent=reasoner_agent)

    # ---- 2) Reason: instruct agent to call tool, then answer
    # We stream to capture tool calls (function_call events) and final answer.
    final_answer = ""
    reasoner_prompt = (
        f"Use search_index(query='{question}', top_k={top_k}) to fetch context.\n"
        f"Then answer the task: {task}\n"
        f"Respond strictly based on retrieved context. If unsupported, say you don't know."
    )

    for ev in reasoner_app.stream_query(user_id=user_id, message=reasoner_prompt):
        # Track tool use
        parts = (ev.get("content") or {}).get("parts", [])
        for p in parts:
            if "function_call" in p:
                trace.append({"step": "retrieve", "tool": p["function_call"].get("name", "search_index"),
                              "args": p["function_call"].get("args", {})})
            if "text" in p:
                final_answer += p["text"]

    final_answer = (final_answer or "").strip()
    trace.append({"step": "reason", "model": reasoner_agent.model, "answer_chars": len(final_answer)})

    # Print a simple trace before the answer (matches your CLI UX)
    print(json.dumps({"trace": trace}, ensure_ascii=False, indent=2))
    return final_answer or "I'm not sure."

if __name__ == "__main__":
    question = ("""Given the following ownership structure: 
                    Co A is 3% owned by SDN A 
                    Co A is 47% owned by Co B
                    Co A is 50% owned by Co C
                    Co B is 42% owned by SDN B
                    Co B is 52% owned by Co D
                    Co D is 50% owned by Person C
                    Co D is 50% owned by SDN D
                    Co C is 52% owned by Trust
                    Co C is 48% owned by Co E
                    Co E is 19% owned by Person E
                    Co E is 81% owned by SDN F
                    Trust is managed by Settlor
                    Trust is managed by Trustee
                    Trust is managed by Beneficiary

                    Further given that Company A Ltd (Co A) is registered in Country X (15% BO threshold),
                    resolve:
                    Task 1: Identify all beneficial owners (FATF Recs 24/25).
                    Task 2: SDN A/B/D/F are on OFAC SDN List. Is Co A subject to US sanctions, and which
                            intermediate/owners contribute? Refer to OFAC FAQ 401 for guidance.
                """)
    answer(question)
