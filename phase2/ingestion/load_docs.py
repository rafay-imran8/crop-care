import os
import json
import fitz  # PyMuPDF

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")        # put PDFs here
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "documents.json")

# -------------------------
# Map PDFs to metadata
# -------------------------
# You can manually add metadata here
PDF_METADATA = {
    "wheat_disease_rust_2011.pdf": {"crop": "wheat", "publication_year": 2011},
    "maize_MLN_disease_2021.pdf": {"crop": "maize", "publication_year": 2021},
    "maize_production_GA_2020.pdf": {"crop": "maize", "publication_year": 2020},
}

# -------------------------
# Ingest PDFs
# -------------------------
documents = []

for filename, meta in PDF_METADATA.items():
    file_path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    pdf_doc = fitz.open(file_path)
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc[page_num]
        text = page.get_text().strip()
        if len(text) == 0:
            continue

        documents.append({
            "text": text,
            "metadata": {
                "crop": meta["crop"],
                "publication_year": meta["publication_year"],
                "source": filename,
                "page": page_num + 1
            }
        })

    pdf_doc.close()
    print(f"Ingested: {filename}")

# -------------------------
# Save to JSON
# -------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"Saved {len(documents)} pages to {OUTPUT_FILE}")
