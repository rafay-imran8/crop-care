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


# --------------------------------------------------
# Extract PDF
# --------------------------------------------------
documents = []

for filename, meta in PDF_METADATA.items():

    file_path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    pdf_doc = fitz.open(file_path)

    extracted_pages = 0
    skipped_pages = 0

    for page_index in range(pdf_doc.page_count):

        page = pdf_doc[page_index]

        raw_text = page.get_text("text")

        if not raw_text.strip():
            continue

        text = clean_text(raw_text)

        if not text:
            continue

        # Skip obvious TOC/front-matter pages.
        if is_likely_toc(text) or is_likely_front_matter(text):
            skipped_pages += 1
            continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "crop": meta["crop"],
                    "publication_year": meta["publication_year"],
                    "source": filename,
                    "page": page_index + 1,
                },
            }
        )

        extracted_pages += 1

    pdf_doc.close()

    print(
        f"Ingested: {filename} | "
        f"pages kept={extracted_pages}, "
        f"pages skipped={skipped_pages}"
    )


# --------------------------------------------------
# Save
# --------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        documents,
        file,
        ensure_ascii=False,
        indent=2,
    )

print(
    f"\nSaved {len(documents)} cleaned pages "
    f"to {OUTPUT_FILE}"
)