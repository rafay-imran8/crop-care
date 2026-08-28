import json
import os
import re


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

DOCS_FILE = os.path.join(PROCESSED_DIR, "documents.json")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "chunks.json")


# --------------------------------------------------
# Chunking Parameters
# --------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 100


# --------------------------------------------------
# Load Documents
# --------------------------------------------------
with open(DOCS_FILE, "r", encoding="utf-8") as file:
    documents = json.load(file)


# --------------------------------------------------
# Text Utilities
# --------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalize whitespace without destroying paragraphs."""

    text = text.replace("\x00", "")
    text = text.replace("\u0001", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_sentences(text: str):
    """
    Simple sentence splitter.

    This intentionally avoids introducing a heavyweight NLP
    dependency for the ingestion pipeline.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def looks_like_heading(text: str) -> bool:
    """
    Detect likely section headings.
    """

    words = text.split()

    if not words:
        return False

    if len(words) > 12:
        return False

    upper_ratio = sum(
        character.isupper()
        for character in text
        if character.isalpha()
    )

    alpha_count = sum(
        character.isalpha()
        for character in text
    )

    if alpha_count == 0:
        return False

    return (
        upper_ratio / alpha_count > 0.65
        or text.endswith(":")
    )


def extract_paragraphs(text: str):
    """
    Split extracted page text into logical paragraphs.
    """

    text = normalize_text(text)

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    return paragraphs


# --------------------------------------------------
# Semantic Chunking
# --------------------------------------------------
def build_chunks_for_document(doc):
    """
    Create semantic chunks from one document/page.

    Chunks are built from sentences rather than arbitrary
    character positions.
    """

    text = normalize_text(doc["text"])
    metadata = doc["metadata"]

    paragraphs = extract_paragraphs(text)

    chunks = []

    current_sentences = []
    current_length = 0

    current_section = None

    for paragraph in paragraphs:

        if looks_like_heading(paragraph):
            current_section = paragraph

            # Keep heading available as context, but don't
            # immediately create a heading-only chunk.
            continue

        sentences = split_sentences(paragraph)

        # If sentence splitting fails, treat the paragraph
        # as one semantic unit.
        if not sentences:
            sentences = [paragraph]

        for sentence in sentences:

            sentence_length = len(sentence)

            # If the sentence itself is too large, split it
            # on word boundaries.
            if sentence_length > CHUNK_SIZE:

                words = sentence.split()

                for word in words:

                    proposed = (
                        f"{' '.join(current_sentences)} {word}"
                    ).strip()

                    if (
                        current_sentences
                        and len(proposed) > CHUNK_SIZE
                    ):
                        chunks.append(
                            {
                                "text": " ".join(
                                    current_sentences
                                ).strip(),
                                "section": current_section,
                            }
                        )

                        # Character-based overlap, but only
                        # between complete words.
                        overlap_words = []
                        overlap_length = 0

                        for previous_word in reversed(
                            current_sentences[-1].split()
                        ):
                            if (
                                overlap_length
                                + len(previous_word)
                                + 1
                                > CHUNK_OVERLAP
                            ):
                                break

                            overlap_words.insert(
                                0,
                                previous_word,
                            )

                            overlap_length += (
                                len(previous_word) + 1
                            )

                        current_sentences = (
                            [" ".join(overlap_words)]
                            if overlap_words
                            else []
                        )

                        current_length = sum(
                            len(item)
                            for item in current_sentences
                        )

                    current_sentences.append(word)
                    current_length += len(word) + 1

                continue

            proposed_length = (
                current_length
                + sentence_length
                + 1
            )

            if (
                current_sentences
                and proposed_length > CHUNK_SIZE
            ):
                chunks.append(
                    {
                        "text": " ".join(
                            current_sentences
                        ).strip(),
                        "section": current_section,
                    }
                )

                # Preserve overlap using complete sentences.
                overlap = []
                overlap_length = 0

                for previous_sentence in reversed(
                    current_sentences
                ):
                    if (
                        overlap_length
                        + len(previous_sentence)
                        + 1
                        > CHUNK_OVERLAP
                    ):
                        break

                    overlap.insert(
                        0,
                        previous_sentence,
                    )

                    overlap_length += (
                        len(previous_sentence) + 1
                    )

                current_sentences = overlap

                current_length = sum(
                    len(item) + 1
                    for item in current_sentences
                )

            current_sentences.append(sentence)
            current_length += sentence_length + 1

    # Flush remaining content.
    if current_sentences:
        chunks.append(
            {
                "text": " ".join(
                    current_sentences
                ).strip(),
                "section": current_section,
            }
        )

    return chunks


# --------------------------------------------------
# Build Chunks
# --------------------------------------------------
all_chunks = []
chunk_id = 0

for doc in documents:

    document_chunks = build_chunks_for_document(doc)

    for item in document_chunks:

        text = item["text"].strip()

        if len(text) < MIN_CHUNK_SIZE:
            continue

        metadata = doc["metadata"].copy()

        metadata["page_start"] = metadata.pop("page")

        metadata["page_end"] = metadata["page_start"]

        if item["section"]:
            metadata["section"] = item["section"]

        all_chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
            }
        )

        chunk_id += 1


# --------------------------------------------------
# Save
# --------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        all_chunks,
        file,
        ensure_ascii=False,
        indent=2,
    )

print(
    f"Saved {len(all_chunks)} semantic chunks "
    f"to {OUTPUT_FILE}"
)