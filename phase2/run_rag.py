from retrieval.retriever import retrieve
from generation.prompt import build_prompt
from generation.generate import generate_answer

SIMILARITY_THRESHOLD = 0.25

def fallback_message():
    return """
I need more information to provide accurate guidance:
• Crop variety?
• Growth stage?
• Region or climate?
• Season or month?
"""

def run_rag(query):
    retrieved = retrieve(query, top_k=5)

    top_score = retrieved[0]["similarity"]

    print(f"\nTop similarity score: {top_score:.3f}\n")

    if top_score < SIMILARITY_THRESHOLD:
        return fallback_message()

    prompt = build_prompt(retrieved, query)
    answer = generate_answer(prompt)

    return answer

if __name__ == "__main__":
    while(True):
        query = input("Enter agriculture question: ")
        response = run_rag(query)

        print("\n--- RAG ANSWER ---\n")
        print(response)
