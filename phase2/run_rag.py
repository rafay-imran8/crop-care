import json
import numpy as np

from retrieval.retriever import retrieve
from generation.generate import generate_answer
from sentence_transformers import SentenceTransformer

# -------------------------
# Config
# -------------------------
CHUNKS_FILE = "embeddings/embed_store.json"
SIMILARITY_THRESHOLD = 0.35
TOP_K = 3

# -------------------------
# Load embedding model
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Load chunks
# -------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_embeddings = np.array([c["embedding"] for c in chunks])

# -------------------------
# Interactive Loop
# -------------------------
# -------------------------
# Batch testing on predefined queries
# -------------------------
# unseen_queries = [
#     # WHEAT
#     "Symptoms of leaf rust in wheat",
#     "Brown rust management in wheat",
#     "Nitrogen deficiency signs in wheat",
#     "Wheat flowering water requirements",
#     "Fungal disease prevention in wheat",
#     "Improving wheat soil fertility",
#     "Best irrigation schedule for wheat",
#     "Using resistant wheat varieties",
#     "Safe pesticide usage in wheat fields",
#     "Monitoring wheat leaves for pests",
    
#     # MAIZE
#     "How to irrigate maize efficiently",
#     "Maize lethal necrosis symptoms",
#     "Water stress effects in maize",
#     "Pest control in maize fields",
#     "Soil nutrient imbalance management for maize",
#     "Caterpillar damage control in maize",
#     "Reducing leaf rolling in maize",
#     "Balanced fertilizer application in maize",
#     "Viral disease detection in maize",
#     "Certified seed usage for maize disease prevention",
    
#     # RICE
#     "Best irrigation schedule for rice",
#     "Managing rice brown spot disease",
#     "Low nitrogen effects on rice growth",
#     "Rice soil fertility improvement techniques",
#     "Controlling rice stem borers",
#     "Water management in rice paddies",
#     "Rice blast disease prevention",
#     "Improving drainage in rice fields",
#     "Organic matter application in rice fields",
#     "Monitoring rice growth under drought"
# ]



for i in range(5):
    query_text = input(f"Enter query {i+1}: ")
    print(f"\nQuery {i}: {query_text}")

    # Encode query
    query_embedding = model.encode(query_text)

    # Retrieve top chunks
    results = retrieve(query_embedding, query_text=query_text, top_k=TOP_K)

    if not results:
        print("\nNo relevant documents found.")
        continue

    top_score = results[0]["similarity"]
    print(f"Top similarity score: {top_score:.3f}")

    # LOW CONFIDENCE FALLBACK
    if top_score < SIMILARITY_THRESHOLD:
        print("I need more information: Crop, stage, region, season")
        continue

    # Generate answer
    answer = generate_answer(results, query_text)
    print("\n--- RAG ANSWER ---\n")
    print(answer)

