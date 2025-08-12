import os, json
import numpy as np
import faiss
from vertexai import init as vertexai_init
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel
from config import PROJECT_ID, LOCATION, TUNING_BASE_MODEL, WORKDIR

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
    vertexai_init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(TUNING_BASE_MODEL)
    ctx = retrieve(q, k=5)
    prompt = (
        "You are an experienced US Sanctions Officer. Answer the question strictly using the provided context. "
        "If the answer is not fully supported by context, say you don't know.\n\n"
        "Context:\n" + "\n---\n".join(ctx) + f"\n\nQuestion: {q}"
    )
    resp = model.generate_content(prompt)
    print(resp.text)

if __name__ == "__main__":
    import sys
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

                    Further given that 𝗖𝗼𝗺𝗽𝗮𝗻𝘆 𝗔 𝗟𝘁𝗱 (𝗖𝗼 𝗔) 𝗶𝘀 𝗿𝗲𝗴𝗶𝘀𝘁𝗲𝗿𝗲𝗱 𝗶𝗻 𝗖𝗼𝘂𝗻𝘁𝗿𝘆 𝗫, 𝘄𝗵𝗶𝗰𝗵 𝗮𝗽𝗽𝗹𝗶𝗲𝘀 𝘁𝗵𝗲 𝟭𝟱% 𝘁𝗵𝗿𝗲𝘀𝗵𝗼𝗹𝗱 𝗳𝗼𝗿 𝗶
                    𝗱𝗲𝗻𝘁𝗶𝗳𝗶𝗰𝗮𝘁𝗶𝗼𝗻 𝗼𝗳 𝗯𝗲𝗻𝗲𝗳𝗶𝗰𝗶𝗮𝗹 𝗼𝘄𝗻𝗲𝗿𝘀𝗵𝗶𝗽, please resolve the two tasks below:

                    𝗧𝗮𝘀𝗸 𝟭 - 𝗕𝗮𝘀𝗲𝗱 𝗼𝗻 𝘁𝗵𝗲 𝗙𝗔𝗧𝗙'𝘀 𝘀𝘁𝗮𝗻𝗱𝗮𝗿𝗱𝘀, 𝗻𝗮𝗺𝗲𝗹𝘆 𝗙𝗔𝗧𝗙 𝗥𝗲𝗰𝗼𝗺𝗺𝗲𝗻𝗱𝗮𝘁𝗶𝗼𝗻𝘀 𝟮𝟰 𝗮𝗻𝗱 𝟮𝟱, 
                    𝗶𝗱𝗲𝗻𝘁𝗶𝗳𝘆 𝗮𝗹𝗹 𝗼𝗳 𝗖𝗼 𝗔's 𝗯𝗲𝗻𝗲𝗳𝗶𝗰𝗶𝗮𝗹 𝗼𝘄𝗻𝗲𝗿𝘀.

                    𝗧𝗮𝘀𝗸 𝟮 - 𝗦𝗗𝗡 𝗔, 𝗦𝗗𝗡 𝗕, 𝗦𝗗𝗡 𝗗 𝗮𝗻𝗱 𝗦𝗗𝗡 𝗙 𝗮𝗿𝗲 𝗮𝗹𝗹 𝗹𝗶𝘀𝘁𝗲𝗱 𝗯𝘆 𝗨𝗦 𝗢𝗙𝗔𝗖 𝗼𝗻 𝗶𝘁𝘀 𝗦𝗽𝗲𝗰𝗶𝗮𝗹𝗹𝘆 𝗗𝗲𝘀𝗶𝗴𝗻𝗮𝘁𝗲𝗱 𝗡𝗮𝘁𝗶𝗼𝗻𝗮𝗹𝘀 𝗟𝗶𝘀𝘁. 
                    𝗜𝘀 𝗖𝗼 𝗔 𝘀𝘂𝗯𝗷𝗲𝗰𝘁 𝘁𝗼 𝗨𝗦 𝘀𝗮𝗻𝗰𝘁𝗶𝗼𝗻𝘀, which of it's intermediate blocking entities direct or indirect owners 
                    contribute to the designation? 𝗣𝗹𝗲𝗮𝘀𝗲 𝗿𝗲𝗳𝗲𝗿 𝘁𝗼 𝗢𝗙𝗔𝗖 𝗙𝗔𝗤 𝟰𝟬𝟭 𝗳𝗼𝗿 𝗴𝘂𝗶𝗱𝗮𝗻𝗰𝗲.
                """)
    answer(question)
