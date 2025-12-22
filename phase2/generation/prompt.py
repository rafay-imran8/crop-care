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

Use ONLY the information provided in the CONTEXT.


CONTEXT:
{context_text}

QUESTION:
{question}

INSTRUCTIONS:
- Answer in 2–3 clear paragraphs
- Cite sources like [1], [2]
- Provide a short Action Checklist (bullets)
- Do NOT include chemical dosage values
- End with a short safety disclaimer

ANSWER:
"""
    return prompt.strip()
