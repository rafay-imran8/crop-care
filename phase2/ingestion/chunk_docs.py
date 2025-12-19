import os
import json

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DOCS_FILE = os.path.join(PROCESSED_DIR, "documents.json")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "chunks.json")

# -------------------------
# Chunking parameters
# -------------------------
CHUNK_SIZE = 500      # number of characters per chunk
CHUNK_OVERLAP = 50    # overlap between chunks

# -------------------------
# Load ingested documents
# -------------------------
with open(DOCS_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

all_chunks = []
chunk_id = 0

for doc in documents:
    text = doc["text"]
    metadata = doc["metadata"]

    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end]

        chunk_meta = metadata.copy()
        chunk_meta["chunk_id"] = chunk_id

        all_chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": chunk_meta
        })

        chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP  # move window forward

# -------------------------
# Save chunked documents
# -------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")
