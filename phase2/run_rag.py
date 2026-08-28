import os

from sentence_transformers import SentenceTransformer

from generation.generate import generate_answer
from retrieval.retriever import retrieve


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.35
TOP_K = 3


# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

print(f"Loading embedding model: {MODEL_NAME}")

embedding_model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# RAG System
# --------------------------------------------------

print("\n" + "=" * 60)
print("Crop Care RAG System")
print("=" * 60)
print("Ask questions about the agricultural knowledge base.")
print("Type 'exit' or 'quit' to stop.\n")


while True:

    # --------------------------------------------------
    # Get User Query
    # --------------------------------------------------

    query_text = input("Enter your query: ").strip()

    if query_text.lower() in {"exit", "quit"}:
        print("\nExiting RAG system.")
        break

    if not query_text:
        print("Please enter a query.\n")
        continue


    # --------------------------------------------------
    # Generate Query Embedding
    # --------------------------------------------------

    query_embedding = embedding_model.encode(
        query_text,
        normalize_embeddings=True,
    )


    # --------------------------------------------------
    # Retrieve Relevant Chunks
    # --------------------------------------------------

    results = retrieve(
        query_embedding=query_embedding,
        query_text=query_text,
        top_k=TOP_K,
    )

    if not results:
        print("\nNo relevant documents found.\n")
        continue


    # --------------------------------------------------
    # Similarity Check
    # --------------------------------------------------

    top_score = results[0]["similarity"]

    print(f"\nTop similarity score: {top_score:.3f}")


    if top_score < SIMILARITY_THRESHOLD:

        print(
            "\nI could not find sufficiently relevant "
            "information in the knowledge base."
        )

        print(
            "Try providing more context such as "
            "crop, disease, growth stage, region, "
            "or season.\n"
        )

        continue


    # --------------------------------------------------
    # Display Retrieved Sources
    # --------------------------------------------------

    print("\n--- Retrieved Sources ---")

    for index, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})

        source = metadata.get(
            "source",
            "Unknown source",
        )

        page = metadata.get("page")

        section = metadata.get("section")

        similarity = result["similarity"]

        print(f"\n[{index}] {source}")

        if page is not None:
            print(f"Page: {page}")

        if section:
            print(f"Section: {section}")

        print(f"Similarity: {similarity:.3f}")


    # --------------------------------------------------
    # Generate RAG Answer
    # --------------------------------------------------

    try:

        answer = generate_answer(
            results,
            query_text,
        )

    except Exception as error:

        print(
            "\nFailed to generate an answer."
        )

        print(f"Error: {error}\n")

        continue


    # --------------------------------------------------
    # Display Answer
    # --------------------------------------------------

    print("\n--- RAG ANSWER ---\n")
    print(answer)

    print("\n" + "-" * 60 + "\n")