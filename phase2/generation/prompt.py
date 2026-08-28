def build_prompt(context_blocks, question):
    """
    Build a grounded prompt for the Crop Care RAG system.

    The model must answer using only the retrieved
    agricultural sources and cite the supporting sources.
    """

    context_parts = []

    for i, chunk in enumerate(context_blocks, start=1):
        metadata = chunk.get("metadata", {})

        source = metadata.get(
            "source",
            "Unknown source",
        )

        year = metadata.get(
            "publication_year",
            "Unknown year",
        )

        page = metadata.get(
            "page",
            "Unknown page",
        )

        context_parts.append(
            f"[{i}] Source: {source} | "
            f"Year: {year} | "
            f"Page: {page}\n"
            f"{chunk.get('text', '').strip()}"
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""
You are Crop Care, an agriculture information assistant.

Your task is to answer the user's question using ONLY the
information contained in the retrieved CONTEXT below.

Do not use outside knowledge, assumptions, or information
that is not supported by the CONTEXT.

CONTEXT:
{context_text}

QUESTION:
{question}

INSTRUCTIONS:

1. Give a clear and concise answer directly addressing the
   question.

2. Ground every factual claim in the provided CONTEXT.

3. Cite the relevant source immediately after the claim using
   the source number, for example [1] or [2].
   If a statement is supported by multiple sources, cite them
   as [1][2].

4. If the CONTEXT does not contain enough information to
   answer the question, explicitly say:
   "The available sources do not provide enough information
   to answer this question."
   Do not guess or fill the gap using outside knowledge.

5. Do not treat information from one crop as information about
   another crop unless the CONTEXT explicitly supports it.

6. Do not provide chemical pesticide or fungicide dosage,
   concentration, application-rate, or mixing instructions.

7. When the sources provide practical recommendations, include
   a short "Action Checklist" with 3–5 concise bullet points.
   Every recommendation must be supported by the CONTEXT.

8. If no practical recommendation is supported by the CONTEXT,
   do not invent an Action Checklist.

9. Keep the response concise and easy for farmers or agricultural
   students to understand.

10. End with a brief safety disclaimer reminding the user to
    follow local agricultural regulations and consult a qualified
    agricultural professional before taking high-risk actions.

RESPONSE FORMAT:

Answer:
[2–3 concise paragraphs with citations]

Action Checklist:
- [Evidence-based action]
- [Evidence-based action]
- [Evidence-based action]

Safety:
[Brief safety disclaimer]
"""

    return prompt.strip()