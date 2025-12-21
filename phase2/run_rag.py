import json
import numpy as np

from retrieval.retriever import retrieve
from generation.generate import generate_answer
from sentence_transformers import SentenceTransformer
CHUNKS_FILE = "embeddings/embed_store.json"
SIMILARITY_THRESHOLD = 0.25
TOP_K = 3
model = SentenceTransformer('all-MiniLM-L6-v2')
# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_embeddings = np.array([c["embedding"] for c in chunks])

query = input("Enter your agriculture question: ")
query = model.encode(query, show_progress_bar=True)
# Retrieve top chunks
results = retrieve(query,top_k=TOP_K)

top_score = results[0]["similarity"]
print(f"\nTop similarity score: {top_score:.3f}")

# LOW CONFIDENCE FALLBACK
if top_score < SIMILARITY_THRESHOLD:
    print("\nI need more information:")
    print("• Crop variety?")
    print("• Growth stage?")
    print("• Region?")
    print("• Season?")
    exit()

# Generate answer
answer = generate_answer(results, query)

print("\n--- RAG ANSWER ---\n")
print(answer)
