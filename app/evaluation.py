from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from app.preprocessing import _load_nlp, _nlp_lock, preprocess_text
from app.nlp import KEYWORD_METHODS, extract_keywords, summarize_abstractive, summarize_extractive


class FrenchTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def summary_scores(
    prediction: str,
    reference: str,
    include_bertscore: bool = False,
) -> dict[str, float]:
    from rouge_score.rouge_scorer import RougeScorer

    scorer = RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=FrenchTokenizer(),
        use_stemmer=False,
    )
    scores = scorer.score(reference, prediction)
    result = {
        f"{metric}_{field}": getattr(value, field)
        for metric, value in scores.items()
        for field in ("precision", "recall", "fmeasure")
    }
    if include_bertscore:
        from bert_score import score

        precision, recall, f1 = score([prediction], [reference], lang="fr", verbose=False)
        result.update(
            bertscore_precision=float(precision[0]),
            bertscore_recall=float(recall[0]),
            bertscore_f1=float(f1[0]),
        )
    return result


def _lemmatize(terms: list[str]) -> set[str]:
    nlp = _load_nlp()
    normalized: set[str] = set()
    for term in terms:
        with _nlp_lock:
            doc = nlp(term)
        key = " ".join(
            token.lemma_.casefold()
            for token in doc
            if token.is_alpha and not token.is_stop
        )
        if key:
            normalized.add(key)
    return normalized


def keyword_scores(
    predicted: list[str | dict[str, Any]],
    reference: list[str | dict[str, Any]],
    k: int = 10,
) -> dict[str, float]:
    def terms(values: list[str | dict[str, Any]]) -> list[str]:
        return [
            str(value.get("terme", "")) if isinstance(value, dict) else value
            for value in values
        ]

    predicted_set = _lemmatize(terms(predicted[:k]))
    reference_set = _lemmatize(terms(reference))
    overlap = len(predicted_set & reference_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(reference_set) if reference_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {f"precision_at_{k}": precision, f"recall_at_{k}": recall, f"f1_at_{k}": f1}


def evaluate_records(
    records: list[dict[str, Any]],
    output_csv: str | Path | None = None,
    include_bertscore: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records:
        prediction = record.get("resume", record.get("summary"))
        reference = record.get("resume_reference", record.get("summary_reference"))
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise ValueError("Chaque entrée doit contenir resume/summary et resume_reference/summary_reference.")
        row: dict[str, Any] = {
            "fichier": record.get("fichier", record.get("filename", "")),
            "methode_resume": record.get("methode_resume", ""),
            "methode_mots_cles": record.get("methode_mots_cles", ""),
        }
        row.update(summary_scores(prediction, reference, include_bertscore))
        row.update(keyword_scores(record.get("mots_cles", []), record.get("mots_cles_reference", []), 5))
        row.update(keyword_scores(record.get("mots_cles", []), record.get("mots_cles_reference", []), 10))
        if "duree_s" in record and "nb_pages" in record:
            row["secondes_par_page"] = record["duree_s"] / max(record["nb_pages"] or 1, 1)
        results.append(row)
    if not results:
        return []
    metrics = [
        key for key, value in results[0].items()
        if isinstance(value, (int, float))
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        key = (row["methode_resume"], row["methode_mots_cles"])
        groups.setdefault(key, []).append(row)
    for (summary_method, keyword_method), rows in groups.items():
        for statistic, compute in (
            ("moyenne", statistics.mean),
            ("ecart_type", lambda values: statistics.stdev(values) if len(values) > 1 else 0.0),
        ):
            aggregate: dict[str, Any] = {
                "fichier": statistic,
                "methode_resume": summary_method,
                "methode_mots_cles": keyword_method,
            }
            aggregate.update({
                metric: compute([float(row[metric]) for row in rows])
                for metric in metrics
            })
            results.append(aggregate)
    if output_csv:
        with Path(output_csv).open("w", encoding="utf-8-sig", newline="") as output:
            fieldnames = list(results[0])
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    return results


def benchmark_records(
    documents: list[dict[str, Any]],
    output_csv: str | Path | None = None,
    include_abstractive: bool = False,
    include_bertscore: bool = False,
) -> list[dict[str, Any]]:
    summary_methods = ["lead3", "tfidf", "textrank", "lexrank", "lsa"]
    if include_abstractive:
        summary_methods.extend(["barthez", "qwen"])
    keyword_methods = list(KEYWORD_METHODS)
    generated: list[dict[str, Any]] = []
    for document in documents:
        text = document.get("texte", document.get("text"))
        reference = document.get("resume_reference", document.get("summary_reference"))
        if not isinstance(text, str) or not isinstance(reference, str):
            raise ValueError("Chaque document doit contenir texte/text et resume_reference/summary_reference.")
        preprocessing_started = time.perf_counter()
        processed = preprocess_text(text)
        preprocessing_duration = time.perf_counter() - preprocessing_started
        if not processed.sentences:
            raise ValueError(f"Aucune phrase exploitable dans {document.get('fichier', 'un document')}.")
        for summary_method in summary_methods:
            summary_started = time.perf_counter()
            if summary_method in {"barthez", "qwen"}:
                summary, _ = summarize_abstractive(
                    processed.text,
                    document.get("longueur", "court"),
                    model_name=summary_method,
                    sentences=processed.sentences,
                )
            else:
                summary = summarize_extractive(
                    summary_method,
                    processed.sentences,
                    processed.normalized_sentences,
                    float(document.get("taux", 0.2)),
                )
            summary_duration = time.perf_counter() - summary_started
            for keyword_method in keyword_methods:
                keyword_started = time.perf_counter()
                keywords = extract_keywords(
                    processed.text,
                    keyword_method,
                    int(document.get("top_k", 10)),
                )
                generated.append({
                    "fichier": document.get("fichier", ""),
                    "methode_resume": summary_method,
                    "methode_mots_cles": keyword_method,
                    "resume": summary,
                    "resume_reference": reference,
                    "mots_cles": keywords,
                    "mots_cles_reference": document.get("mots_cles_reference", []),
                    "duree_s": preprocessing_duration + summary_duration + time.perf_counter() - keyword_started,
                    "nb_pages": document.get("nb_pages") or 1,
                })
    return evaluate_records(generated, output_csv, include_bertscore)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue les résumés et mots-clés d'un corpus annoté.")
    parser.add_argument("input_json", help="Fichier JSON contenant une liste d'objets à évaluer.")
    parser.add_argument("output_csv", help="Chemin du tableau de résultats CSV.")
    parser.add_argument("--bertscore", action="store_true", help="Calcule aussi BERTScore (téléchargement du modèle requis).")
    parser.add_argument("--benchmark", action="store_true", help="Lance toutes les méthodes sur le texte et les références fournis.")
    parser.add_argument("--include-abstractive", action="store_true", help="Ajoute BARThez et Qwen au benchmark (lent et modèle à télécharger).")
    args = parser.parse_args()
    records = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Le JSON d'entrée doit contenir une liste d'objets.")
    if args.benchmark:
        benchmark_records(records, args.output_csv, args.include_abstractive, args.bertscore)
    else:
        evaluate_records(records, args.output_csv, args.bertscore)


if __name__ == "__main__":
    main()
