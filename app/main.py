from __future__ import annotations

import os
import re
import tempfile
import time
from io import BytesIO
from pathlib import Path

from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import BASE_DIR
from app.evaluation import evaluate_records
from app.nlp import KEYWORD_METHODS, SUMMARY_METHODS
from app.local_model import BARTHEZ_MODEL_NAME, MODEL_NAME, barthez_status, model_status
from app.pipeline import run_pipeline
from app.preprocessing import FrenchModelUnavailable
from app.summarizer import extract_document

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "250"))
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
LENGTHS = {
    "court": {"label": "Synthétique", "guide": "5 à 7 phrases, environ 130 à 180 mots.", "tokens": 240},
    "moyen": {"label": "Équilibré", "guide": "environ 300 à 450 mots.", "tokens": 600},
    "detaille": {"label": "Approfondi", "guide": "environ 700 à 900 mots.", "tokens": 1100},
}
app = FastAPI(title="Clair — résumés locaux")


class KeywordExport(BaseModel):
    terme: str
    score: float


class DocxExport(BaseModel):
    filename: str
    summary: str
    pages: int | None = None
    model: str = "Qwen 2.5 0.5B (local)"
    keywords: list[KeywordExport] = Field(default_factory=list)


@app.get("/")
async def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health", include_in_schema=False)
@app.get("/api/health")
async def health():
    import importlib.util
    import shutil

    try:
        import spacy

        french_model_installed = spacy.util.is_package("fr_core_news_md")
    except ImportError:
        french_model_installed = False
    keybert_installed = all(
        importlib.util.find_spec(module) is not None
        for module in ("keybert", "sentence_transformers")
    )
    return {
        "mode": "local",
        "qwen": {"model": MODEL_NAME, **model_status()},
        "barthez": {"model": BARTHEZ_MODEL_NAME, **barthez_status()},
        "summary_methods": {**SUMMARY_METHODS, "qwen": f"{MODEL_NAME} (local)"},
        "keyword_methods": KEYWORD_METHODS,
        "dependencies_installed": all(
            importlib.util.find_spec(module) is not None
            for module in ("pymupdf", "spacy", "sklearn", "networkx", "yake", "langdetect")
        ),
        "french_model_installed": french_model_installed,
        "keybert_installed": keybert_installed,
        "ocr_available": shutil.which("tesseract") is not None,
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.post("/api/summarize")
async def summarize(
    file: UploadFile = File(...),
    method: str = Form("textrank"),
    rate: int = Form(20),
    keyword_method: str = Form("yake"),
    top_k: int = Form(10),
    length: str = Form("court"),
    force_ocr: bool = Form(False),
):
    started = time.perf_counter()
    filename = file.filename or "document"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Formats acceptés : PDF, Word .docx, texte .txt et Markdown .md.")
    if method not in SUMMARY_METHODS:
        raise HTTPException(status_code=422, detail="Méthode de résumé non reconnue.")
    if keyword_method not in KEYWORD_METHODS:
        raise HTTPException(status_code=422, detail="Méthode d'extraction des mots-clés non reconnue.")
    if not 10 <= rate <= 40:
        raise HTTPException(status_code=422, detail="Le taux doit être compris entre 10 et 40 %.")
    if not 1 <= top_k <= 30:
        raise HTTPException(status_code=422, detail="Le nombre de mots-clés doit être compris entre 1 et 30.")
    if length not in LENGTHS:
        raise HTTPException(status_code=422, detail="Longueur de résumé non reconnue.")
    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Fichier trop volumineux (limite actuelle : {MAX_UPLOAD_MB} Mo).")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(data)
        temp_path = Path(temp_file.name)
    try:
        extracted = await run_in_threadpool(extract_document, temp_path, force_ocr)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"Impossible de lire ce document : {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Une dépendance d'extraction est absente.") from exc
    finally:
        temp_path.unlink(missing_ok=True)
    text = extracted.text
    if len(text.strip()) < 100:
        raise HTTPException(status_code=422, detail="Aucun texte exploitable après l'extraction et l'OCR éventuelle.")
    try:
        result = await run_in_threadpool(
            run_pipeline,
            filename,
            extracted,
            method,
            rate,
            keyword_method,
            top_k,
            length,
        )
    except FrenchModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Une dépendance NLP est absente. Installez les dépendances du projet.",
        ) from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"Impossible de charger un modèle local : {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result["model"] = f"{MODEL_NAME} (local)" if method == "qwen" else SUMMARY_METHODS[method]
    result["duree_s"] = round(time.perf_counter() - started, 3)
    return result


@app.post("/api/evaluate")
async def evaluate(payload: list[dict], bertscore: bool = False):
    try:
        return await run_in_threadpool(evaluate_records, payload, None, bertscore)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Données d'évaluation invalides : {exc}") from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Une dépendance d'évaluation est absente.") from exc


@app.post("/api/export/docx")
async def export_docx(payload: DocxExport):
    document = Document()
    document.core_properties.title = f"Résumé — {Path(payload.filename).stem}"
    document.add_heading("Résumé du document", level=0)
    document.add_paragraph(Path(payload.filename).name, style="Subtitle")
    details = [payload.model]
    if payload.pages:
        details.append(f"{payload.pages} pages")
    document.add_paragraph(" · ".join(details))
    for paragraph in re.split(r"\n\s*\n", payload.summary.strip()):
        if paragraph.strip():
            document.add_paragraph(paragraph.strip())
    if payload.keywords:
        document.add_heading("Mots-clés", level=1)
        document.add_paragraph(", ".join(item.terme for item in payload.keywords))
    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)
    safe_stem = re.sub(r"[^\w-]+", "-", Path(payload.filename).stem, flags=re.UNICODE).strip("-") or "document"
    return StreamingResponse(buffer, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="resume-{safe_stem}.docx"'})

