import os
import sys
import json
import math
import numpy as np

from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Add phase2 to Python path
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR = os.path.dirname(CURRENT_DIR)

if PHASE2_DIR not in sys.path:
    sys.path.insert(0, PHASE2_DIR)

from retrieval.retriever import retrieve
from generation.generate import generate_answer


# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K_VALUES = [1, 3, 5]
DEFAULT_K = 5

# Set to True if you want to evaluate LLM answers too.
EVALUATE_GENERATION = True


# --------------------------------------------------
# Paths
# --------------------------------------------------
EMBED_STORE_FILE = os.path.join(
    PHASE2_DIR,
    "embeddings",
    "embed_store.json",
)

QUERIES_FILE = os.path.join(
    CURRENT_DIR,
    "queries.json",
)

OUTPUT_FILE = os.path.join(
    CURRENT_DIR,
    "rag_evaluation_results.json",
)


# --------------------------------------------------
# Load Data
# --------------------------------------------------
print("Loading embedding store...")

with open(
    EMBED_STORE_FILE,
    "r",
    encoding="utf-8",
) as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


print("Loading evaluation queries...")

with open(
    QUERIES_FILE,
    "r",
    encoding="utf-8",
) as file:
    queries = json.load(file)

print(f"Loaded {len(queries)} evaluation queries.")


# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------
print(f"Loading embedding model: {MODEL_NAME}")

embedding_model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def reciprocal_rank(relevant, retrieved):
    """
    Calculate Reciprocal Rank.

    Returns:
        1/rank of first relevant result,
        or 0 if no relevant result is found.
    """

    for rank, item in enumerate(retrieved, start=1):

        if item in relevant:
            return 1.0 / rank

    return 0.0


def dcg(relevance_scores):
    """
    Calculate Discounted Cumulative Gain.
    """

    score = 0.0

    for rank, relevance in enumerate(
        relevance_scores,
        start=1,
    ):
        score += relevance / math.log2(rank + 1)

    return score


def ndcg_at_k(relevant, retrieved, k):
    """
    Calculate NDCG@K.

    Binary relevance is used:
        1 = relevant
        0 = not relevant
    """

    retrieved_k = retrieved[:k]

    relevance_scores = [
        1 if item in relevant else 0
        for item in retrieved_k
    ]

    actual_dcg = dcg(relevance_scores)

    ideal_length = min(
        len(relevant),
        k,
    )

    ideal_scores = [1] * ideal_length

    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def precision_at_k(relevant, retrieved, k):
    """
    Precision@K.
    """

    retrieved_k = retrieved[:k]

    if not retrieved_k:
        return 0.0

    relevant_count = sum(
        item in relevant
        for item in retrieved_k
    )

    return relevant_count / len(retrieved_k)


def recall_at_k(relevant, retrieved, k):
    """
    Recall@K.
    """

    if not relevant:
        return 0.0

    retrieved_k = retrieved[:k]

    relevant_count = sum(
        item in relevant
        for item in retrieved_k
    )

    return relevant_count / len(relevant)


def hit_rate_at_k(relevant, retrieved, k):
    """
    Hit Rate@K.

    Returns 1 if at least one relevant
    document is retrieved.
    """

    retrieved_k = retrieved[:k]

    return float(
        any(
            item in relevant
            for item in retrieved_k
        )
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
all_results = []

aggregate = {
    k: {
        "precision": [],
        "recall": [],
        "hit_rate": [],
        "ndcg": [],
    }
    for k in TOP_K_VALUES
}

mrr_scores = []

generation_results = []


for query_number, query_data in enumerate(
    queries,
    start=1,
):

    query_text = query_data["query"]

    # --------------------------------------------------
    # Ground Truth
    # --------------------------------------------------
    relevant_docs = set(
        query_data["relevant_docs"]
    )

    print(
        f"\n[{query_number}/{len(queries)}] "
        f"{query_text}"
    )

    # --------------------------------------------------
    # Query Embedding
    # --------------------------------------------------
    query_embedding = embedding_model.encode(
        query_text,
        normalize_embeddings=True,
    )

    # --------------------------------------------------
    # Retrieve
    # --------------------------------------------------
    retrieved_chunks = retrieve(
        query_embedding,
        query_text=query_text,
        top_k=DEFAULT_K,
    )

    # --------------------------------------------------
    # Convert retrieved chunks to document sources
    # --------------------------------------------------
    retrieved_docs = [
        chunk["metadata"].get(
            "source",
            "unknown",
        )
        for chunk in retrieved_chunks
    ]

    # --------------------------------------------------
    # Remove duplicate sources while preserving ranking
    # --------------------------------------------------
    unique_retrieved_docs = []

    for source in retrieved_docs:

        if source not in unique_retrieved_docs:
            unique_retrieved_docs.append(source)

    # --------------------------------------------------
    # Retrieval Metrics
    # --------------------------------------------------
    query_metrics = {}

    for k in TOP_K_VALUES:

        precision = precision_at_k(
            relevant_docs,
            unique_retrieved_docs,
            k,
        )

        recall = recall_at_k(
            relevant_docs,
            unique_retrieved_docs,
            k,
        )

        hit_rate = hit_rate_at_k(
            relevant_docs,
            unique_retrieved_docs,
            k,
        )

        ndcg = ndcg_at_k(
            relevant_docs,
            unique_retrieved_docs,
            k,
        )

        aggregate[k]["precision"].append(
            precision
        )

        aggregate[k]["recall"].append(
            recall
        )

        aggregate[k]["hit_rate"].append(
            hit_rate
        )

        aggregate[k]["ndcg"].append(
            ndcg
        )

        query_metrics[f"precision@{k}"] = precision
        query_metrics[f"recall@{k}"] = recall
        query_metrics[f"hit_rate@{k}"] = hit_rate
        query_metrics[f"ndcg@{k}"] = ndcg

    # --------------------------------------------------
    # MRR
    # --------------------------------------------------
    rr = reciprocal_rank(
        relevant_docs,
        unique_retrieved_docs,
    )

    mrr_scores.append(rr)

    query_metrics["reciprocal_rank"] = rr

    # --------------------------------------------------
    # Similarity
    # --------------------------------------------------
    if retrieved_chunks:

        query_metrics["top_similarity"] = (
            retrieved_chunks[0]["similarity"]
        )

        query_metrics["average_similarity"] = (
            sum(
                chunk["similarity"]
                for chunk in retrieved_chunks
            )
            / len(retrieved_chunks)
        )

    else:

        query_metrics["top_similarity"] = 0.0
        query_metrics["average_similarity"] = 0.0

    # --------------------------------------------------
    # Generation Evaluation
    # --------------------------------------------------
    generated_answer = None
    has_citations = False

    if EVALUATE_GENERATION and retrieved_chunks:

        try:

            generated_answer = generate_answer(
                retrieved_chunks,
                query_text,
            )

            # Check whether answer contains
            # at least one source citation.
            has_citations = any(
                f"[{i}]"
                in generated_answer
                for i in range(
                    1,
                    len(retrieved_chunks) + 1,
                )
            )

            generation_results.append({
                "has_answer": True,
                "has_citations": has_citations,
            })

        except Exception as error:

            print(
                f"Generation failed: {error}"
            )

            generated_answer = None

            generation_results.append({
                "has_answer": False,
                "has_citations": False,
            })

    # --------------------------------------------------
    # Save Per Query Result
    # --------------------------------------------------
    all_results.append({
        "query": query_text,
        "relevant_docs": list(relevant_docs),
        "retrieved_docs": unique_retrieved_docs,
        "retrieved_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["metadata"].get(
                    "source",
                    "unknown",
                ),
                "page": chunk["metadata"].get(
                    "page",
                    None,
                ),
                "similarity": chunk["similarity"],
            }
            for chunk in retrieved_chunks
        ],
        "metrics": query_metrics,
        "generated_answer": generated_answer,
        "has_citations": has_citations,
    })


# --------------------------------------------------
# Aggregate Metrics
# --------------------------------------------------
summary = {}

for k in TOP_K_VALUES:

    summary[f"precision@{k}"] = float(
        np.mean(
            aggregate[k]["precision"]
        )
    )

    summary[f"recall@{k}"] = float(
        np.mean(
            aggregate[k]["recall"]
        )
    )

    summary[f"hit_rate@{k}"] = float(
        np.mean(
            aggregate[k]["hit_rate"]
        )
    )

    summary[f"ndcg@{k}"] = float(
        np.mean(
            aggregate[k]["ndcg"]
        )
    )


summary["mrr"] = float(
    np.mean(mrr_scores)
)


# --------------------------------------------------
# Generation Summary
# --------------------------------------------------
if generation_results:

    total_generation = len(
        generation_results
    )

    successful_answers = sum(
        result["has_answer"]
        for result in generation_results
    )

    cited_answers = sum(
        result["has_citations"]
        for result in generation_results
    )

    summary["answer_generation_success_rate"] = (
        successful_answers / total_generation
    )

    summary["citation_rate"] = (
        cited_answers / total_generation
    )


# --------------------------------------------------
# Final Output
# --------------------------------------------------
evaluation_output = {
    "configuration": {
        "embedding_model": MODEL_NAME,
        "num_queries": len(queries),
        "num_chunks": len(chunks),
        "top_k_values": TOP_K_VALUES,
        "generation_evaluation": EVALUATE_GENERATION,
    },
    "summary": summary,
    "queries": all_results,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        evaluation_output,
        file,
        indent=2,
        ensure_ascii=False,
    )


# --------------------------------------------------
# Print Summary
# --------------------------------------------------
print("\n")
print("=" * 60)
print("RAG EVALUATION SUMMARY")
print("=" * 60)

print(
    f"Queries evaluated : {len(queries)}"
)

print(
    f"Chunks available  : {len(chunks)}"
)

print()

for k in TOP_K_VALUES:

    print(
        f"--- K = {k} ---"
    )

    print(
        f"Precision@{k}: "
        f"{summary[f'precision@{k}']:.4f}"
    )

    print(
        f"Recall@{k}:    "
        f"{summary[f'recall@{k}']:.4f}"
    )

    print(
        f"Hit Rate@{k}:  "
        f"{summary[f'hit_rate@{k}']:.4f}"
    )

    print(
        f"NDCG@{k}:      "
        f"{summary[f'ndcg@{k}']:.4f}"
    )

    print()


print(
    f"MRR: "
    f"{summary['mrr']:.4f}"
)

if "answer_generation_success_rate" in summary:

    print(
        f"Answer generation success: "
        f"{summary['answer_generation_success_rate']:.4f}"
    )

    print(
        f"Citation rate: "
        f"{summary['citation_rate']:.4f}"
    )

print()

print(
    f"Detailed results saved to:\n"
    f"{OUTPUT_FILE}"
)

print("=" * 60)