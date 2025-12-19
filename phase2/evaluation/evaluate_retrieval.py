import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR,"embeddings")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

os.makedirs(EVAL_DIR, exist_ok=True)

CHUNKS_FILE = os.path.join(PROCESSED_DIR, "embed_store.json")
QUERIES_FILE = os.path.join(EVAL_DIR, "queries.json")
OUTPUT_FILE = os.path.join(EVAL_DIR, "retrieval_metrics.json")

TOP_K = 1  # Top-k retrieval

# -------------------------
# Load chunks and queries
# -------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries = json.load(f)

# Load embeddings as numpy array
chunk_embeddings = np.array([c["embedding"] for c in chunks])  # shape: (num_chunks, embedding_dim)

print(f"Loaded {len(chunks)} chunks, embedding dim={chunk_embeddings.shape[1]}")

# -------------------------
# Load sentence transformer
# -------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # same model as before

# -------------------------
# Evaluate retrieval
# -------------------------
results = []

for q in queries:
    query_text = q["query"]
    relevant_docs = set(q["relevant_docs"])

    # 1. Compute embedding for the query
    query_embedding = embed_model.encode([query_text])[0]  # shape: (embedding_dim,)

    # 2. Compute cosine similarity
    # cosine_similarity = dot(A,B)/(||A||*||B||)
    norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    sims = (chunk_embeddings @ query_embedding) / norms  # shape: (num_chunks,)
    
    # 3. Get top-k indices
    topk_indices = sims.argsort()[-TOP_K:][::-1]

    # 4. Compute Precision@k and Recall@k
    retrieved_docs = set([chunks[i]["metadata"]["source"] for i in topk_indices])
    num_relevant_retrieved = len(retrieved_docs & relevant_docs)

    precision = num_relevant_retrieved / TOP_K
    recall = num_relevant_retrieved / len(relevant_docs)

    # 5. Save result for this query
    results.append({
        "query": query_text,
        "precision@5": precision,
        "recall@5": recall,
        "retrieved_docs": list(retrieved_docs),
        "relevant_docs": list(relevant_docs)
    })

# -------------------------
# Save results
# -------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Saved retrieval evaluation to {OUTPUT_FILE}")
