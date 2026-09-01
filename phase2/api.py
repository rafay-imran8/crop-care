from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app_service import service


FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
app = FastAPI(title="Crop Care API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=8)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents")
def documents():
    return {"documents": service.documents()}


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    crop: str = Form(...),
    publication_year: int = Form(...),
):
    try:
        record = service.ingest(await file.read(), file.filename or "", crop, publication_year)
        return {"document": record}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {error}") from error


@app.post("/api/chat")
def chat(request: AskRequest):
    try:
        return service.ask(request.question, request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {error}") from error


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/")
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")
