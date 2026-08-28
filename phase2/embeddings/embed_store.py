import json
import os

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "chunks.json",
)

EMBEDDINGS_DIR = os.path.join(
    BASE_DIR,
    "embeddings",
)

EMBED_STORE_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "embed_store.json",
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------
# Load Chunks
# --------------------------------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise ValueError("No chunks found. Run load.py and chunk.py first.")


texts = [chunk["text"] for chunk in chunks]


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------
print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# Generate Embeddings
# --------------------------------------------------
print(f"Embedding {len(texts)} chunks...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True,
)


# --------------------------------------------------
# Build Embedding Store
# --------------------------------------------------
embed_store = []

for chunk, embedding in zip(chunks, embeddings):

    embed_store.append(
        {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "embedding": embedding.tolist(),
            "metadata": chunk["metadata"],
        }
    )


# --------------------------------------------------
# Save
# --------------------------------------------------
os.makedirs(
    EMBEDDINGS_DIR,
    exist_ok=True,
)

with open(
    EMBED_STORE_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        embed_store,
        file,
        ensure_ascii=False,
        indent=2,
    )

print(
    f"Saved embeddings for {len(embed_store)} chunks "
    f"to {EMBED_STORE_FILE}"
)

print(
    f"Embedding dimension: {len(embed_store[0]['embedding'])}"
)