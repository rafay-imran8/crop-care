import json
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Paths
# -------------------------
EMBED_STORE_FILE = "embeddings/embed_store.json"

# -------------------------
# Load embeddings
# -------------------------
with open(EMBED_STORE_FILE, "r", encoding="utf-8") as f:
    embed_store = json.load(f)

# Convert embeddings to numpy array for fast similarity computation
all_embeddings = np.array([c["embedding"] for c in embed_store])
print(f"Loaded {all_embeddings.shape[0]} embeddings, dim={all_embeddings.shape[1]}")

# -------------------------
# Load embedding model for queries
# -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# Retrieval function
# -------------------------
def retrieve(query, top_k=5, crop_filter=None):
    """
    query: string, e.g., "wheat rust symptoms"
    top_k: number of chunks to return
    crop_filter: optional filter, e.g., "wheat" or "maize"
    """
    # Step 1: embed query
    query_emb = model.encode([query])[0]  # shape (384,)
    
    # Step 2: compute cosine similarity
    # cosine_sim = dot(a,b) / (||a||*||b||)
    query_norm = np.linalg.norm(query_emb)
    embeddings_norm = np.linalg.norm(all_embeddings, axis=1)
    cosine_sim = np.dot(all_embeddings, query_emb) / (embeddings_norm * query_norm + 1e-8)
    
    # Step 3: apply metadata filter
    filtered_indices = list(range(len(embed_store)))
    if crop_filter:
        filtered_indices = [i for i in range(len(embed_store)) if embed_store[i]["metadata"]["crop"] == crop_filter]
        cosine_sim = cosine_sim[filtered_indices]
    
    # Step 4: get top-k indices
    top_indices = np.argsort(-cosine_sim)[:top_k]
    if crop_filter:
        top_indices = [filtered_indices[i] for i in top_indices]
    
    # Step 5: return chunks
    results = []
    for idx in top_indices:
        results.append({
            "text": embed_store[idx]["metadata"]["source"] + ": " + embed_store[idx]["text"][:200] + "...",
            "metadata": embed_store[idx]["metadata"],
            "similarity": float(cosine_sim[idx])
        })
    return results

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    query = "wheat rust symptoms"
    results = retrieve(query, top_k=3, crop_filter="wheat")
    for i, r in enumerate(results):
        print(f"\nRank {i+1}, Similarity: {r['similarity']:.3f}")
        print("Source:", r["metadata"]["source"])
        print("Text snippet:", r["text"])
