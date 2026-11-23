from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any


MODEL_NAME = "fr_core_news_md"
_nlp_lock = threading.Lock()
_nlp: Any = None


class FrenchModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PreprocessedDocument:
    text: str
    sentences: list[str]
    normalized_sentences: list[str]
    language: str | None
    warnings: list[str]


def _load_nlp() -> Any:
    global _nlp
    with _nlp_lock:
        if _nlp is not None:
            return _nlp
        try:
            import spacy

            _nlp = spacy.load(MODEL_NAME, disable=["ner"])
        except (ImportError, OSError) as exc:
            raise FrenchModelUnavailable(
                "Le modèle spaCy français est absent. Installez-le avec "
                "`python -m spacy download fr_core_news_md`."
            ) from exc
        return _nlp


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"(?<![.!?:])\n(?!\s*\n)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def preprocess_text(text: str) -> PreprocessedDocument:
    cleaned = clean_text(text)
    warnings: list[str] = []
    language: str | None = None
    if len(cleaned) >= 40:
        try:
            from langdetect import DetectorFactory, detect
            from langdetect.lang_detect_exception import LangDetectException

            DetectorFactory.seed = 0
            language = detect(cleaned)
            if language != "fr":
                warnings.append(
                    f"Langue détectée : {language}; les résultats sont optimisés pour le français."
                )
        except ImportError as exc:
            raise FrenchModelUnavailable(
                "La détection de langue nécessite langdetect. Installez les dépendances du projet."
            ) from exc
        except LangDetectException:
            warnings.append("La langue du document n'a pas pu être déterminée.")

    nlp = _load_nlp()
    with _nlp_lock:
        if len(cleaned) >= nlp.max_length:
            nlp.max_length = len(cleaned) + 1
        doc = nlp(cleaned)
    sentences: list[str] = []
    normalized: list[str] = []
    for sentence in doc.sents:
        raw = sentence.text.strip()
        words = [token for token in sentence if not token.is_space and not token.is_punct]
        if len(words) < 5:
            continue
        if words and sum(token.like_num for token in words) / len(words) >= 0.6:
            continue
        norm = " ".join(
            token.lemma_.casefold()
            for token in words
            if token.is_alpha and not token.is_stop and token.lemma_.strip()
        )
        if norm:
            sentences.append(raw)
            normalized.append(norm)

    if not sentences:
        warnings.append(
            "Aucune phrase exploitable après filtrage; le texte extrait reste disponible."
        )
    if re.search(r"\w-\s*\n\s*\w", text):
        warnings.append(
            "Des mots coupés en fin de ligne ont été recollés; vérifiez les mots composés."
        )
    return PreprocessedDocument(cleaned, sentences, normalized, language, warnings)
