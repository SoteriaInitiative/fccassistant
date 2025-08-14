import os, json, re
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



ROUTER_MODEL_NAME = os.getenv("ROUTER_MODEL_NAME",
                              f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-2.5-flash",)
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", TUNING_BASE_MODEL)
aiplatform.init(project=PROJECT_ID, location=LOCATION)
endpoints = aiplatform.Endpoint.list(
    filter=f'display_name="{TUNING_DISPLAY_NAME}"',
    order_by="update_time desc"
)
TUNED_MODEL_NAME = os.getenv("TUNED_MODEL_NAME",
                             f"projects/{PROJECT_ID}/locations/us-central1/endpoints/{endpoints[0].name}")

GENERAL_MODEL_NAME = os.getenv("GENERAL_MODEL_NAME",
                               f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-2.0-flash",)
GENERAL_INSTRUCTION = (
    "You do not answer the question. Kindly decline to analyse the user question and explain you are not build for general use."
)

SANCTIONS_INSTRUCTION = (
    "You are an experienced US Sanctions Officer. "
    "Use ONLY the context returned by the search_index tool. "
    "If the answer is not fully supported by context, say you don't know."
)

DOMAIN_PROFILES = {
    "sanctions": {
        "model": lambda use_tuned: (TUNED_MODEL_NAME if use_tuned else BASE_MODEL_NAME),
        "instruction": SANCTIONS_INSTRUCTION,
        "use_tool": True,   # use retrieval
    },
    "general": {
        "model": lambda _: GENERAL_MODEL_NAME,
        "instruction": GENERAL_INSTRUCTION,
        "use_tool": False,  # no retrieval by default
    },
}

def _extract_first_json(s: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", s)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}

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
            "You are a router. Classify the user's question and plan to solve the question.\n"
            "ONLY set the domain='sanctions' if the user's question concerns sanctions.\n"
            "A question is clearly sanctions related if includes key words such as OFAC, sanctions, SND.\n"
            "designated person, HMT, SECO, .\n"
            "Return ONLY strict JSON with keys:\n"
            "  domain: 'sanctions' | 'general'\n"
            "  top_k: integer 5..15 \n"
            "  use_tuned: bool (only meaningful for 'sanctions')\n"
            "  task: one concise instruction to the solver"
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



# ----------------------------
# Orchestration with AdkApp + stream_query
# ----------------------------
def run_with_adk(question: str, user_id: str = "local-user") -> str:
    vertexai_init(project=PROJECT_ID, location=LOCATION)

    router_app = AdkApp(agent=build_router_agent())

    trace: List[Dict[str, Any]] = []

    # ---- 1) Route: get plan JSON ----
    router_text = ""
    for ev in router_app.stream_query(user_id=user_id, message=f"User question:\n{question}\n\nJSON only:"):
        if ev.get("content") and ev["content"].get("parts"):
            router_text += "".join(p.get("text", "") for p in ev["content"]["parts"])

    try:
        plan = _extract_first_json(router_text)
    except Exception:
        plan = {}
        trace.append({"step": "route-creation-exception", "data": {
            "router_text": router_text,
        }})

    domain = (plan.get("domain") or "general").strip().lower()
    # allow 0 → skip retrieval
    try:
        top_k = int(plan.get("top_k", 5))
    except Exception:
        top_k = 5
    top_k = max(0, min(top_k, 10))
    use_tuned = bool(plan.get("use_tuned", domain == "sanctions"))
    task = (plan.get("task") or question).strip()

    # Build domain reasoner
    reasoner_agent = build_reasoner_agent_for(domain, use_tuned)
    reasoner_app = AdkApp(agent=reasoner_agent)

    trace.append({"step": "route-end", "data": {
        "domain": domain, "top_k": top_k, "use_tuned": use_tuned, "task": task,
        "model": reasoner_agent.model
    }})

    # ---- 2) Reason: tool (if any) → final answer ----
    # ---- 2) Reason: tool (if any) → final answer ----
    if domain == "general":
        # Temporary block for general domain reasoning
        final_answer = (
            "I can’t process non-sanctions questions at the moment. "
            "This assistant is restricted to sanctions-related queries."
        )
        trace.append({"step": "general-block", "reason": "blocked by policy"})
    else:
        # Proceed with the normal reasoning flow for sanctions
        if DOMAIN_PROFILES.get(domain, {}).get("use_tool") and top_k > 0:
            reasoner_prompt = (
                f"Use search_index(query='{question}', top_k={top_k}) to fetch context.\n"
                f"Then answer the task: {task}\n"
                f"Respond strictly based on retrieved context. If unsupported, say you don't know."
            )
        else:
            reasoner_prompt = (
                f"Task: {task}\n"
                f"Answer helpfully and concisely. If information is uncertain, state that explicitly."
            )

        final_answer = ""
        for ev in reasoner_app.stream_query(user_id=user_id, message=reasoner_prompt):
            parts = (ev.get("content") or {}).get("parts", [])
            for p in parts:
                if "function_call" in p:
                    trace.append({
                        "step": "retrieve",
                        "tool": p["function_call"].get("name", "search_index"),
                        "args": p["function_call"].get("args", {})
                    })
                if "text" in p:
                    final_answer += p["text"]

    final_answer = (final_answer or "").strip()
    trace.append({"step": "reason", "model": reasoner_agent.model, "answer_chars": len(final_answer)})

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
