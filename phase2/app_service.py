import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from embeddings.embed_store import (
    embed_chunks,
    load_store as load_embedding_store,
    save_store,
)
from generation.generate import generate_answer
from ingestion.chunk_docs import build_chunks
from ingestion.load_docs import extract_pdf
from retrieval.retriever import load_store as load_retrieval_store
from retrieval.retriever import retrieve_from_store


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CATALOG_FILE = BASE_DIR / "data" / "processed" / "catalog.json"
EMBED_STORE_FILE = BASE_DIR / "embeddings" / "embed_store.json"
MODEL_NAME = "all-MiniLM-L6-v2"


class CropCareService:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None

    def _embedding_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def _catalog(self):
        if not CATALOG_FILE.exists():
            return []
        return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))

    def _save_catalog(self, catalog):
        CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_FILE.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def documents(self):
        catalog = self._catalog()
        existing = {item["filename"]: item for item in catalog}
        for item in load_embedding_store(str(EMBED_STORE_FILE)):
            metadata = item.get("metadata", {})
            source = metadata.get("source")
            if source and source not in existing:
                existing[source] = {
                    "filename": source,
                    "crop": metadata.get("crop", "unknown"),
                    "publication_year": metadata.get("publication_year"),
                    "pages": None,
                    "chunks": 0,
                    "status": "ready",
                }
                existing[source]["chunks"] = 0
            if source and existing[source].get("pages") is None:
                existing[source]["chunks"] += 1
        return list(existing.values())

    def ingest(self, file_bytes, filename, crop, publication_year):
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF documents are supported.")
        if not file_bytes:
            raise ValueError("The uploaded PDF is empty.")
        if not crop.strip():
            raise ValueError("Crop is required.")
        if publication_year < 1000 or publication_year > datetime.now().year:
            raise ValueError("Publication year is outside the valid range.")

        with self._lock:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = Path(filename).name
            destination = RAW_DIR / safe_name
            destination.write_bytes(file_bytes)
            try:
                documents = extract_pdf(
                    str(destination), safe_name, crop.strip().lower(), publication_year
                )
            except Exception as error:
                destination.unlink(missing_ok=True)
                raise ValueError(f"Could not read the PDF: {error}") from error
            if not documents:
                destination.unlink(missing_ok=True)
                raise ValueError("No readable agricultural text was found in the PDF.")

            current_store = load_embedding_store(str(EMBED_STORE_FILE))
            next_id = max((item["chunk_id"] for item in current_store), default=-1) + 1
            new_chunks = build_chunks(documents, starting_id=next_id)
            new_embeddings = embed_chunks(new_chunks, model=self._embedding_model())
            save_store(current_store + new_embeddings, str(EMBED_STORE_FILE))

            catalog = self.documents()
            record = {
                "filename": safe_name,
                "crop": crop.strip().lower(),
                "publication_year": publication_year,
                "pages": len(documents),
                "chunks": len(new_chunks),
                "status": "ready",
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            catalog = [item for item in catalog if item["filename"] != safe_name]
            catalog.insert(0, record)
            self._save_catalog(catalog)
            return record

    def ask(self, question, top_k=3):
        question = question.strip()
        if not question:
            raise ValueError("Question is required.")
        if len(question) > 1000:
            raise ValueError("Question must be 1000 characters or fewer.")
        model = self._embedding_model()
        query_embedding = model.encode(question, normalize_embeddings=True)
        store, embeddings = load_retrieval_store(str(EMBED_STORE_FILE))
        sources = retrieve_from_store(
            query_embedding, query_text=question, top_k=top_k,
            store_chunks=store, store_embeddings=embeddings,
        )
        if not sources:
            return {"answer": "I could not find relevant information in the uploaded sources.", "sources": []}
        if sources[0]["similarity"] < 0.35:
            return {"answer": "The available sources do not provide enough information to answer this question.", "sources": sources}
        return {"answer": generate_answer(sources, question), "sources": sources}


service = CropCareService()
