import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "embed_store.json",
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation",
)

QUERIES_FILE = os.path.join(
    EVALUATION_DIR,
    "queries.json",
)

OUTPUT_FILE = os.path.join(
    EVALUATION_DIR,
    "retrieval_metrics.json",
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


# --------------------------------------------------
# Load Data
# --------------------------------------------------
with open(
    EMBEDDINGS_FILE,
    "r",
    encoding="utf-8",
) as file:
    chunks = json.load(file)


with open(
    QUERIES_FILE,
    "r",
    encoding="utf-8",
) as file:
    queries = json.load(file)


if not chunks:
    raise ValueError(
        "No embedded chunks found."
    )


# --------------------------------------------------
# Embeddings
# --------------------------------------------------
chunk_embeddings = np.asarray(
    [
        chunk["embedding"]
        for chunk in chunks
    ],
    dtype=np.float32,
)

# Ensure stored vectors are normalized.
norms = np.linalg.norm(
    chunk_embeddings,
    axis=1,
    keepdims=True,
)

norms[norms == 0] = 1

chunk_embeddings = (
    chunk_embeddings / norms
)


# --------------------------------------------------
# Model
# --------------------------------------------------
print(
    f"Loading embedding model: {MODEL_NAME}"
)

embed_model = SentenceTransformer(
    MODEL_NAME
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
results = []

total_precision = 0.0
total_recall = 0.0


for query in queries:

    query_text = query["query"]

    # --------------------------------------------------
    # Ground Truth
    # --------------------------------------------------
    relevant_chunks = set(
        query.get(
            "relevant_chunks",
            []
        )
    )

    relevant_documents = set(
        query.get(
            "relevant_docs",
            []
        )
    )

    # --------------------------------------------------
    # Query Embedding
    # --------------------------------------------------
    query_embedding = embed_model.encode(
        query_text,
        normalize_embeddings=True,
    )

    # --------------------------------------------------
    # Similarity
    # --------------------------------------------------
    similarities = (
        chunk_embeddings
        @ query_embedding
    )

    # --------------------------------------------------
    # Retrieve Top-K
    # --------------------------------------------------
    k = min(
        TOP_K,
        len(chunks),
    )

    topk_indices = np.argsort(
        -similarities
    )[:k]

    retrieved_chunks = [
        chunks[index]
        for index in topk_indices
    ]

    retrieved_chunk_ids = {
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    }

    retrieved_documents = {
        chunk["metadata"]["source"]
        for chunk in retrieved_chunks
    }

    # --------------------------------------------------
    # Chunk-Level Evaluation
    # --------------------------------------------------
    if relevant_chunks:

        relevant_retrieved = (
            retrieved_chunk_ids
            & relevant_chunks
        )

        precision = (
            len(relevant_retrieved)
            / k
        )

        recall = (
            len(relevant_retrieved)
            / len(relevant_chunks)
        )

        evaluation_level = "chunk"

    # --------------------------------------------------
    # Document-Level Backward Compatibility
    # --------------------------------------------------
    elif relevant_documents:

        relevant_retrieved = (
            retrieved_documents
            & relevant_documents
        )

        precision = (
            len(relevant_retrieved)
            / k
        )

        recall = (
            len(relevant_retrieved)
            / len(relevant_documents)
        )

        evaluation_level = "document"

    else:

        precision = 0.0
        recall = 0.0
        evaluation_level = "undefined"

    total_precision += precision
    total_recall += recall

    results.append(
        {
            "query": query_text,
            "evaluation_level": evaluation_level,
            "precision@5": precision,
            "recall@5": recall,
            "retrieved_chunks": [
                chunk["chunk_id"]
                for chunk in retrieved_chunks
            ],
            "retrieved_documents": list(
                retrieved_documents
            ),
            "relevant_chunks": list(
                relevant_chunks
            ),
            "relevant_documents": list(
                relevant_documents
            ),
        }
    )


# --------------------------------------------------
# Aggregate Metrics
# --------------------------------------------------
num_queries = len(queries)

if num_queries:
    mean_precision = (
        total_precision / num_queries
    )

    mean_recall = (
        total_recall / num_queries
    )
else:
    mean_precision = 0.0
    mean_recall = 0.0


output = {
    "top_k": TOP_K,
    "num_queries": num_queries,
    "mean_precision@5": mean_precision,
    "mean_recall@5": mean_recall,
    "queries": results,
}


# --------------------------------------------------
# Save
# --------------------------------------------------
os.makedirs(
    EVALUATION_DIR,
    exist_ok=True,
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        output,
        file,
        ensure_ascii=False,
        indent=2,
    )


print(
    f"\nEvaluated {num_queries} queries."
)

print(
    f"Mean Precision@5: "
    f"{mean_precision:.4f}"
)

print(
    f"Mean Recall@5: "
    f"{mean_recall:.4f}"
)

print(
    f"Saved evaluation to: "
    f"{OUTPUT_FILE}"
)