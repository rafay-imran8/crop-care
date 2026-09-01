import json
import os
import re

import fitz  # PyMuPDF


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "documents.json")


# --------------------------------------------------
# PDF Metadata
# --------------------------------------------------
PDF_METADATA = {
    "wheat_disease_rust_2011.pdf": {
        "crop": "wheat",
        "publication_year": 2011,
    },
    "maize_MLN_disease_2021.pdf": {
        "crop": "maize",
        "publication_year": 2021,
    },
    "maize_production_GA_2020.pdf": {
        "crop": "maize",
        "publication_year": 2020,
    },
    "rice_good_practice_2008.pdf": {
        "crop": "rice",
        "publication_year": 2008,
    },
}


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------
def clean_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts while preserving
    meaningful text and paragraph boundaries.
    """

    # Remove NULL and other problematic control characters.
    text = text.replace("\x00", "")
    text = text.replace("\u0001", "")
    text = text.replace("\ufeff", "")

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces around newlines.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix words broken across lines by PDF extraction.
    # Example:
    # "agricul-\n ture" -> "agriculture"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Convert remaining single newlines within a paragraph to spaces.
    # Paragraph boundaries represented by blank lines are preserved.
    paragraphs = []

    for paragraph in text.split("\n\n"):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if paragraph:
            paragraphs.append(paragraph)

    return "\n\n".join(paragraphs).strip()


# --------------------------------------------------
# Page Quality Checks
# --------------------------------------------------
def is_likely_toc(text: str) -> bool:
    """
    Detect pages that appear to be tables of contents.

    This is intentionally conservative so that useful content
    is not accidentally removed.
    """

    lower = text.lower()

    toc_markers = [
        "contents",
        "table of contents",
    ]

    if any(marker in lower for marker in toc_markers):
        return True

    # A page containing many short heading-like lines and
    # page-number patterns is likely a TOC.
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 8:
        return False

    page_number_lines = sum(
        bool(re.search(r"\b\d{1,3}\s*$", line))
        for line in lines
    )

    short_lines = sum(
        len(line.split()) <= 8
        for line in lines
    )

    return (
        page_number_lines >= 4
        and short_lines / len(lines) > 0.6
    )


def is_likely_front_matter(text: str) -> bool:
    """
    Detect obvious publication front matter such as author,
    copyright, design, and funding pages.
    """

    lower = text.lower()

    markers = [
        "copyright",
        "design assistants",
        "cover photos",
        "internationally funded",
        "responsibility for this publication",
    ]

    matches = sum(marker in lower for marker in markers)

    return matches >= 2


def extract_pdf(file_path, filename=None, crop="unknown", publication_year=None):
    """Extract usable pages from one PDF with caller-provided metadata."""
    filename = filename or os.path.basename(file_path)
    documents = []

    with fitz.open(file_path) as pdf_doc:
        for page_index in range(pdf_doc.page_count):
            text = clean_text(pdf_doc[page_index].get_text("text"))
            if not text or is_likely_toc(text) or is_likely_front_matter(text):
                continue

            documents.append({
                "text": text,
                "metadata": {
                    "crop": crop,
                    "publication_year": publication_year,
                    "source": filename,
                    "page": page_index + 1,
                },
            })

    return documents


def extract_directory(metadata_by_filename=None):
    """Extract all configured PDFs in the legacy raw-data directory."""
    metadata_by_filename = metadata_by_filename or PDF_METADATA
    documents = []
    for filename, meta in metadata_by_filename.items():
        file_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        extracted = extract_pdf(file_path, filename, **meta)
        documents.extend(extracted)
        print(f"Ingested: {filename} | pages kept={len(extracted)}")
    return documents


def save_documents(documents, output_file=OUTPUT_FILE):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(documents, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    documents = extract_directory()
    save_documents(documents)
    print(f"\nSaved {len(documents)} cleaned pages to {OUTPUT_FILE}")