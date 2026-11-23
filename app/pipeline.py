from __future__ import annotations

import time
from typing import Any

from app.nlp import extract_keywords, summarize_abstractive, summarize_extractive
from app.preprocessing import preprocess_text
from app.summarizer import ExtractionResult


def run_pipeline(
    filename: str,
    extracted: ExtractionResult,
    method: str = "textrank",
    rate: int = 20,
    keyword_method: str = "yake",
    top_k: int = 10,
    length: str = "court",
) -> dict[str, Any]:
    started = time.perf_counter()
    processed = preprocess_text(extracted.text)
    if not processed.sentences:
        raise ValueError("Le document ne contient pas assez de phrases exploitables.")

    if method in {"qwen", "barthez"}:
        summary, chunks = summarize_abstractive(
            processed.text,
            length,
            model_name=method,
            sentences=processed.sentences,
        )
    else:
        summary = summarize_extractive(
            method,
            processed.sentences,
            processed.normalized_sentences,
            rate / 100,
        )
        chunks = len(processed.sentences)
    keywords = extract_keywords(processed.text, keyword_method, top_k)

    source_words = len(extracted.text.split())
    summary_words = len(summary.split())
    reduction = min(99.9, max(0, round((1 - summary_words / max(source_words, 1)) * 100, 1)))
    return {
        "fichier": filename,
        "filename": filename,
        "langue": processed.language,
        "nb_pages": extracted.pages,
        "pages": extracted.pages,
        "characters": len(extracted.text),
        "words": source_words,
        "summary_words": summary_words,
        "reduction_percent": reduction,
        "chunks": chunks,
        "methode_resume": method,
        "taux": rate / 100,
        "methode_mots_cles": keyword_method,
        "mots_cles": keywords,
        "duree_s": round(time.perf_counter() - started, 3),
        "avertissements": extracted.warnings + processed.warnings,
        "resume": summary,
        "summary": summary,
        "texte_extrait": extracted.text,
        "texte_pretraite": processed.text,
        "phrases_analysees": len(processed.sentences),
    }
