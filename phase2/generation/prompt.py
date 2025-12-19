def build_prompt(context_blocks, question):
    context_text = ""

    for i, c in enumerate(context_blocks, start=1):
        source = c["metadata"].get("source", "unknown")
        year = c["metadata"].get("publication_year", "unknown")

        context_text += (
            f"[{i}] {source} ({year})\n"
            f"{c['text']}\n\n"
        )

    prompt = f"""
You are an agriculture extension expert.

Use ONLY the information provided in the context to answer the question.
Do NOT use outside knowledge.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""
    return prompt.strip()
