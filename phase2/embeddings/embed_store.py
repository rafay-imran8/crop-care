import json
import os
import torch
from sentence_transformers import SentenceTransformer

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "processed", "chunks.json")
EMBED_STORE_FILE = os.path.join(BASE_DIR, "embeddings", "embed_store.json")

# -------------------------
# Load chunks
# -------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

# -------------------------
# Load embedding model
# -------------------------
# Using a small, efficient sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings
# Returns a tensor of shape (num_chunks, embedding_dim)
embeddings = model.encode(texts, show_progress_bar=True)

# -------------------------
# Save embeddings along with metadata
# -------------------------
embed_store = []
for i, c in enumerate(chunks):
    embed_store.append({
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        "embedding": embeddings[i].tolist(),  # convert to list for JSON
        "metadata": c["metadata"]
    })

os.makedirs(os.path.dirname(EMBED_STORE_FILE), exist_ok=True)
with open(EMBED_STORE_FILE, "w", encoding="utf-8") as f:
    json.dump(embed_store, f, ensure_ascii=False, indent=2)

print(f"Saved embeddings for {len(embed_store)} chunks to {EMBED_STORE_FILE}")
