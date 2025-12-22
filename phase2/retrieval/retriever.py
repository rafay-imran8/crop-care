# retrieval/retriever.py

import json
import numpy as np

# -------------------------
# Load chunks with embeddings
# -------------------------
CHUNKS_FILE = "embeddings/embed_store.json"

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_embeddings = np.array([c["embedding"] for c in chunks])

# -------------------------
# Crop Detection (Query Routing)
# -------------------------
def detect_crop(query: str):
    query = query.lower()
    if "wheat" in query:
        return "wheat"
    if "maize" in query or "corn" in query:
        return "maize"
    if "rice" in query:
        return "rice"
    return None

# -------------------------
# Chunk Filtering
# -------------------------
def filter_chunks(chunks, min_len=150):
    """
    Remove low-quality chunks such as references, figures, authors, etc.
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

# -------------------------
# Retrieval Function (with routing)
# -------------------------
def retrieve(query_embedding, query_text=None, top_k=3):
    """
    Compute cosine similarity and return top_k chunks.
    If query_text is provided, retrieval is routed by crop.
    """

    # --- STEP 1: Crop-based routing ---
    if query_text:
        crop = detect_crop(query_text)
        if crop:
            routed = [
                (c, emb)
                for c, emb in zip(chunks, chunk_embeddings)
                if c["metadata"].get("crop") == crop
            ]
        else:
            routed = list(zip(chunks, chunk_embeddings))
    else:
        routed = list(zip(chunks, chunk_embeddings))

    if len(routed) == 0:
        return []

    routed_chunks, routed_embeddings = zip(*routed)
    routed_embeddings = np.vstack(routed_embeddings)

    # --- STEP 2: Cosine similarity ---
    similarities = routed_embeddings @ query_embedding.T
    if similarities.ndim == 2:
        similarities = similarities.squeeze()

    idx_sorted = np.argsort(-similarities)[:top_k]

    top_chunks = []
    for i in idx_sorted:
        chunk = routed_chunks[i].copy()
        chunk["similarity"] = float(similarities[i])
        top_chunks.append(chunk)

    # --- STEP 3: Quality filtering ---
    top_chunks = filter_chunks(top_chunks)

    return top_chunks
