import json
import numpy as np

from retrieval.retriever import retrieve
from generation.generate import generate_answer
from sentence_transformers import SentenceTransformer

# -------------------------
# Config
# -------------------------
CHUNKS_FILE = "embeddings/embed_store.json"
SIMILARITY_THRESHOLD = 0.35
TOP_K = 3

# -------------------------
# Load embedding model
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Load chunks
# -------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_embeddings = np.array([c["embedding"] for c in chunks])

# -------------------------
# Interactive Loop
# -------------------------
i = 0
while i < 5:
    query_text = input("\nEnter your agriculture question: ").strip()

    # Encode query (KEEP BOTH)
    query_embedding = model.encode(query_text)

    # Retrieve top chunks
    results = retrieve(
        query_embedding,
        query_text=query_text,
        top_k=TOP_K
    )

    if not results:
        print("\nNo relevant documents found.")
        i += 1
        continue

    top_score = results[0]["similarity"]
    print(f"\nTop similarity score: {top_score:.3f}")

    # -------------------------
    # LOW CONFIDENCE FALLBACK
    # -------------------------
    if top_score < SIMILARITY_THRESHOLD:
        print("\nI need more information:")
        print("• Crop variety?")
        print("• Growth stage?")
        print("• Region?")
        print("• Season?")
        i += 1
        continue

    # -------------------------
    # Generate answer
    # -------------------------
    answer = generate_answer(results, query_text)

    print("\n--- RAG ANSWER ---\n")
    print(answer)

    i += 1
