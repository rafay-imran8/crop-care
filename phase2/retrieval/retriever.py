import json
import os

import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

EMBED_STORE_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "embed_store.json",
)


# --------------------------------------------------
# Load Embedding Store
# --------------------------------------------------
with open(
    EMBED_STORE_FILE,
    "r",
    encoding="utf-8",
) as file:
    chunks = json.load(file)


if not chunks:
    raise ValueError(
        "Embedding store is empty."
    )


chunk_embeddings = np.asarray(
    [chunk["embedding"] for chunk in chunks],
    dtype=np.float32,
)


# --------------------------------------------------
# Crop Detection
# --------------------------------------------------
def detect_crop(query: str):
    """
    Detect the crop explicitly mentioned in a query.
    """

    query = query.lower()

    crop_aliases = {
        "wheat": ["wheat"],
        "maize": ["maize", "corn"],
        "rice": ["rice"],
    }

    for crop, aliases in crop_aliases.items():

        for alias in aliases:

            if alias in query:
                return crop

    return None


# --------------------------------------------------
# Quality Filtering
# --------------------------------------------------
def is_quality_chunk(chunk, min_len=150):
    """
    Determine whether a chunk is suitable for retrieval.
    """

    text = chunk.get("text", "").strip().lower()

    if len(text) < min_len:
        return False

    bad_markers = [
        "references",
        "copyright",
        "table of contents",
        "design assistants",
        "cover photos",
    ]

    if any(marker in text for marker in bad_markers):
        return False

    # Reject chunks that are mostly non-alphabetic content.
    alpha_chars = sum(
        character.isalpha()
        for character in text
    )

    if len(text) > 0:
        alpha_ratio = alpha_chars / len(text)

        if alpha_ratio < 0.45:
            return False

    return True


def filter_chunks(chunks):
    """
    Filter low-quality chunks before similarity search.
    """

    return [
        chunk
        for chunk in chunks
        if is_quality_chunk(chunk)
    ]


# --------------------------------------------------
# Retrieval
# --------------------------------------------------
def retrieve(
    query_embedding,
    query_text=None,
    top_k=3,
):
    """
    Retrieve the most relevant chunks using cosine similarity.

    Query routing is applied when the crop can be identified.
    """

    if top_k <= 0:
        return []

    # --------------------------------------------------
    # Normalize query embedding
    # --------------------------------------------------
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        return []

    query_embedding = (
        query_embedding / query_norm
    )

    # --------------------------------------------------
    # Quality filtering BEFORE retrieval
    # --------------------------------------------------
    candidate_chunks = filter_chunks(chunks)

    if not candidate_chunks:
        return []

    candidate_ids = {
        chunk["chunk_id"]
        for chunk in candidate_chunks
    }

    candidate_indices = [
        index
        for index, chunk in enumerate(chunks)
        if chunk["chunk_id"] in candidate_ids
    ]

    candidate_embeddings = chunk_embeddings[
        candidate_indices
    ]

    # --------------------------------------------------
    # Crop routing
    # --------------------------------------------------
    crop = None

    if query_text:
        crop = detect_crop(query_text)

    if crop:
        crop_mask = np.array(
            [
                chunks[index]["metadata"].get("crop")
                == crop
                for index in candidate_indices
            ]
        )

        if crop_mask.any():

            candidate_indices = [
                index
                for index, include in zip(
                    candidate_indices,
                    crop_mask,
                )
                if include
            ]

            candidate_embeddings = chunk_embeddings[
                candidate_indices
            ]

    # --------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------
    similarities = candidate_embeddings @ query_embedding

    # --------------------------------------------------
    # Top-k
    # --------------------------------------------------
    top_k = min(
        top_k,
        len(similarities),
    )

    top_indices = np.argsort(
        -similarities
    )[:top_k]

    results = []

    for index in top_indices:

        original_index = candidate_indices[index]

        result = chunks[
            original_index
        ].copy()

        result["similarity"] = float(
            similarities[index]
        )

        results.append(result)

    return results