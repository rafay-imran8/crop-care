import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "processed", "chunks.json")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
EMBED_STORE_FILE = os.path.join(EMBEDDINGS_DIR, "embed_store.json")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_store(store_file=EMBED_STORE_FILE):
    if not os.path.exists(store_file):
        return []
    with open(store_file, "r", encoding="utf-8") as file:
        return json.load(file)


def embed_chunks(chunks, model=None):
    if not chunks:
        return []
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "embedding": embedding.tolist(),
            "metadata": chunk["metadata"],
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]


def save_store(embed_store, store_file=EMBED_STORE_FILE):
    os.makedirs(os.path.dirname(store_file), exist_ok=True)
    with open(store_file, "w", encoding="utf-8") as file:
        json.dump(embed_store, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    if not chunks:
        raise ValueError("No chunks found. Run load_docs.py and chunk_docs.py first.")
    print(f"Loading embedding model: {MODEL_NAME}")
    embed_store = embed_chunks(chunks)
    save_store(embed_store)
    print(f"Saved embeddings for {len(embed_store)} chunks to {EMBED_STORE_FILE}")
    print(f"Embedding dimension: {len(embed_store[0]['embedding'])}")