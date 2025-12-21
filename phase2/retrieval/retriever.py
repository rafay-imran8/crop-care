# retrieval/retriever.py

import json
import numpy as np

# Load chunks with embeddings
CHUNKS_FILE = "embeddings/embed_store.json"
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_embeddings = np.array([c["embedding"] for c in chunks])

def retrieve(query_embedding, top_k=3):
    """
    Compute cosine similarity and return top_k chunks
    """
    # Cosine similarity
    similarities = chunk_embeddings @ query_embedding.T
    if similarities.ndim == 2:
        similarities = similarities.squeeze()

    idx_sorted = np.argsort(-similarities)[:top_k]
    top_chunks = []
    for i in idx_sorted:
        chunk = chunks[i].copy()
        chunk["similarity"] = float(similarities[i])
        top_chunks.append(chunk)
    return top_chunks

def filter_chunks(chunks, min_len=150):
    """
    Filter chunks to remove references, figures, authors, etc.
    """
    filtered = []
    for c in chunks:
        text = c["text"].lower()
        if (
            len(text) >= min_len
            and "references" not in text
            and "figure" not in text
            and "authors" not in text
            and "copyright" not in text
        ):
            filtered.append(c)
    return filtered
