def build_prompt(retrieved_chunks, user_query):
    """
    retrieved_chunks: list of dicts with keys:
        - text
        - metadata {source, year}
    """

    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk["metadata"].get("source", "Unknown")
        year = chunk["metadata"].get("year", "N/A")

        context_blocks.append(
            f"[{i}] Source: {source} | Year: {year}\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are an agriculture assistant.

Context:
{context}

Question:
{user_query}

Instructions:
- Answer ONLY using the context above.
- Cite sources using [1], [2], etc.
- Provide a short actionable checklist.
- Avoid chemical dosage instructions.
- If the context is insufficient, ask clarifying questions.
- Include a brief safety disclaimer.
"""

    return prompt.strip()
