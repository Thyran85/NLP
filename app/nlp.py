from __future__ import annotations

import math
import os
import re
import threading
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import CACHE_DIR
from app.preprocessing import _load_nlp, _nlp_lock
from app.summarizer import split_text

SUMMARY_METHODS = {
    "lead3": "Lead-3",
    "tfidf": "TF-IDF + position",
    "textrank": "TextRank",
    "lsa": "LSA",
    "lexrank": "LexRank",
    "barthez": "BARThez (moussaKam/barthez-orangesum-abstract)",
    "qwen": "Qwen2.5-0.5B-Instruct (local)",
}
KEYWORD_METHODS = {
    "yake": "YAKE",
    "keybert": "KeyBERT",
}
_keybert_lock = threading.Lock()
_keybert_instance: Any = None


def _sentence_count(total: int, ratio: float) -> int:
    if not 0.1 <= ratio <= 0.4:
        raise ValueError("Le taux de résumé doit être compris entre 0,1 et 0,4.")
    return min(total, max(1, math.ceil(total * ratio)))


def summarize_extractive(
    method: str,
    sentences: list[str],
    normalized_sentences: list[str],
    ratio: float = 0.2,
) -> str:
    if method not in {"lead3", "tfidf", "textrank", "lexrank", "lsa"}:
        raise ValueError(f"Méthode extractive inconnue : {method}.")
    if not sentences:
        raise ValueError("Aucune phrase exploitable pour produire un résumé.")
    if method == "lead3":
        return " ".join(sentences[:min(3, len(sentences))])
    if len(sentences) != len(normalized_sentences):
        raise ValueError("Les phrases brutes et normalisées doivent être alignées.")
    count = _sentence_count(len(sentences), ratio)
    if len(sentences) == 1:
        return sentences[0]

    documents = [text if text.strip() else "texte" for text in normalized_sentences]
    matrix = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(documents)
    if method == "tfidf":
        scores = np.asarray(matrix.sum(axis=1)).ravel()
        positions = np.linspace(1.0, 0.0, len(sentences), endpoint=True)
        scores += positions * max(float(scores.max()), 1.0) * 0.1
    elif method in {"textrank", "lexrank"}:
        similarities = cosine_similarity(matrix)
        np.fill_diagonal(similarities, 0.0)
        if method == "lexrank":
            similarities[similarities < 0.1] = 0.0
            scores = similarities.sum(axis=1)
            if not np.any(scores):
                similarities = cosine_similarity(matrix)
                np.fill_diagonal(similarities, 0.0)
                scores = similarities.sum(axis=1)
        else:
            graph = nx.from_numpy_array(similarities)
            scores_by_index = nx.pagerank(graph, weight="weight")
            scores = np.array([scores_by_index[index] for index in range(len(sentences))])
    else:
        component_count = min(5, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if component_count < 1:
            scores = np.asarray(matrix.sum(axis=1)).ravel()
        else:
            reduced = TruncatedSVD(n_components=component_count, random_state=0).fit_transform(matrix)
            scores = np.linalg.norm(reduced, axis=1)

    selected = sorted(np.argsort(-scores, kind="stable")[:count].tolist())
    return " ".join(sentences[index] for index in selected)


def _pack_sentences(sentences: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_size = 0
            chunks.extend(split_text(sentence, max_chars))
            continue
        if current and current_size + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = []
            current_size = 0
        current.append(sentence)
        current_size += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_abstractive(
    text: str,
    length: str,
    model_name: str = "qwen",
    sentences: list[str] | None = None,
) -> tuple[str, int]:
    if model_name == "qwen":
        from app.local_model import summarize_with_qwen

        generate = summarize_with_qwen
        chunk_chars = int(os.getenv("CHUNK_CHARS", "36000"))
    elif model_name == "barthez":
        from app.local_model import summarize_with_barthez

        generate = summarize_with_barthez
        chunk_chars = 2700
    else:
        raise ValueError(f"Modèle abstractif inconnu : {model_name}.")

    targets = {
        "court": {"guide": "5 à 7 phrases, environ 130 à 180 mots.", "tokens": 240},
        "moyen": {"guide": "environ 300 à 450 mots.", "tokens": 600},
        "detaille": {"guide": "environ 700 à 900 mots.", "tokens": 1100},
    }
    if length not in targets:
        raise ValueError("Longueur de résumé non reconnue.")
    if model_name == "barthez" and sentences:
        parts = _pack_sentences(sentences, chunk_chars)
    else:
        parts = split_text(text, chunk_chars)
    briefs = [
        generate(
            part,
            f"Résume fidèlement cette partie en français en 100 à 140 mots. "
            f"Explique ses idées et leurs liens. Partie {index} sur {len(parts)}; "
            "ne conclus pas sur l'ensemble du document.",
            max_new_tokens=220,
        )
        for index, part in enumerate(parts, start=1)
    ]
    notes = "\n\n".join(briefs)
    rounds = 0
    while len(notes) > chunk_chars and rounds < 8:
        groups = split_text(notes, chunk_chars)
        condensed = "\n\n".join(
            generate(
                group,
                "Fusionne ces résumés en une synthèse fidèle, en reliant les idées et les étapes.",
                max_new_tokens=480 if model_name == "qwen" else 400,
            )
            for group in groups
        )
        if len(condensed) >= len(notes):
            raise RuntimeError("Le modèle n'a pas suffisamment condensé les résumés intermédiaires.")
        notes = condensed
        rounds += 1
    if len(notes) > chunk_chars:
        raise RuntimeError("Les résumés intermédiaires dépassent encore la limite de contexte du modèle.")
    target = targets[length]
    result = generate(
        notes,
        f"Rédige le résumé final en français. Longueur : {target['guide']} "
        "Présente le sujet, le fil directeur et les idées majeures dans leur contexte. "
        "Écris une prose autonome, sans liste. N'invente rien.",
        max_new_tokens=target["tokens"],
    )
    return result, len(parts)


def _keybert_model() -> Any:
    global _keybert_instance
    with _keybert_lock:
        if _keybert_instance is not None:
            return _keybert_instance
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder=str(CACHE_DIR),
        )
        _keybert_instance = KeyBERT(model=encoder)
        return _keybert_instance


def _normalized_phrase(phrase: str, stop_words: set[str]) -> tuple[str, str] | None:
    nlp = _load_nlp()
    with _nlp_lock:
        doc = nlp(phrase)
    tokens = [token for token in doc if token.is_alpha]
    while tokens and tokens[0].lemma_.casefold() in stop_words:
        tokens.pop(0)
    while tokens and tokens[-1].lemma_.casefold() in stop_words:
        tokens.pop()
    if not tokens:
        return None
    display = " ".join(token.text for token in tokens)
    key = " ".join(token.lemma_.casefold() for token in tokens)
    return display, key


def extract_keywords(text: str, method: str, top_k: int = 10) -> list[dict[str, float | str]]:
    if method not in KEYWORD_METHODS:
        raise ValueError(f"Méthode de mots-clés inconnue : {method}.")
    if not 1 <= top_k <= 30:
        raise ValueError("Le nombre de mots-clés doit être compris entre 1 et 30.")

    nlp = _load_nlp()
    stop_words = {
        token
        for word in nlp.Defaults.stop_words
        for token in re.findall(r"(?u)\b\w\w+\b", word.casefold())
    }
    if method == "yake":
        import yake

        candidates = [
            (phrase, 1.0 / (1.0 + max(float(score), 0.0)))
            for phrase, score in yake.KeywordExtractor(
                lan="fr", n=3, top=top_k * 3, dedupLim=0.9
            ).extract_keywords(text)
        ]
    else:
        model = _keybert_model()
        chunks = split_text(text, 800)
        candidates: list[tuple[str, float]] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            candidates.extend(
                (phrase, max(0.0, min(1.0, float(score))))
                for phrase, score in model.extract_keywords(
                    chunk,
                    keyphrase_ngram_range=(1, 3),
                    stop_words=sorted(stop_words),
                    use_mmr=True,
                    diversity=0.5,
                    top_n=top_k,
                )
            )

    best: dict[str, tuple[str, float]] = defaultdict(lambda: ("", 0.0))
    for phrase, score in candidates:
        normalized = _normalized_phrase(phrase, stop_words)
        if not normalized:
            continue
        display, key = normalized
        if not key:
            continue
        if score > best[key][1]:
            best[key] = (display, score)
    ranked = sorted(best.values(), key=lambda item: item[1], reverse=True)[:top_k]
    return [{"terme": phrase, "score": round(score, 4)} for phrase, score in ranked]
