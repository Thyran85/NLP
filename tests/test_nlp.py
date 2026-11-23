from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from app import evaluation, pipeline, preprocessing
from app.nlp import summarize_abstractive, summarize_extractive
from app.preprocessing import PreprocessedDocument
from app.summarizer import ExtractionResult, _remove_repeated_margins, extract_document, split_text


def test_clean_text_normalizes_ligatures_hyphenation_and_urls():
    cleaned = preprocessing.clean_text("ﬁle https://example.test/x\ninter-\n national\ntexte")
    assert cleaned == "file international texte"


def test_remove_repeated_headers_and_page_numbers():
    pages = [
        ["Rapport 2024", "Texte A", "Page 1"],
        ["Rapport 2025", "Texte B", "Page 2"],
        ["Rapport 2026", "Texte C", "Page 3"],
    ]
    assert _remove_repeated_margins(pages) == [["Texte A"], ["Texte B"], ["Texte C"]]


def test_extract_pdf_orders_columns_and_reports_page_count(tmp_path: Path):
    pdf_path = tmp_path / "columns.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((40, 60), "Gauche alpha beta gamma delta")
    page.insert_text((330, 60), "Droite alpha beta gamma delta")
    document.save(pdf_path)
    document.close()

    extracted = extract_document(pdf_path)

    assert extracted.pages == 1
    assert extracted.text.index("Gauche alpha") < extracted.text.index("Droite alpha")
    assert extracted.warnings == []


def test_protected_pdf_returns_clear_error(tmp_path: Path):
    pdf_path = tmp_path / "protected.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(
        pdf_path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="secret",
    )
    document.close()

    with pytest.raises(ValueError, match="protégé par mot de passe"):
        extract_document(pdf_path)


def test_scanned_pdf_without_tesseract_reports_installation_error(tmp_path: Path, monkeypatch):
    from app import summarizer

    pdf_path = tmp_path / "scanned.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    def no_tesseract(name):
        assert name == "tesseract"
        return None

    monkeypatch.setattr(summarizer.shutil, "which", no_tesseract)

    with pytest.raises(RuntimeError, match="Tesseract est requis"):
        extract_document(pdf_path)


def test_ocr_extracts_scanned_page_and_adds_warning(tmp_path: Path, monkeypatch):
    import pytesseract

    from app import summarizer

    pdf_path = tmp_path / "scanned.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    monkeypatch.setattr(summarizer.shutil, "which", lambda name: "/usr/bin/tesseract" if name == "tesseract" else None)
    def fake_ocr(image, lang):
        assert image.width > 0
        assert lang == "fra"
        return "Texte reconnu par OCR française."

    monkeypatch.setattr(pytesseract, "image_to_string", fake_ocr)

    extracted = extract_document(pdf_path)

    assert extracted.text == "Texte reconnu par OCR française."
    assert extracted.warnings == ["Page 1 traitée par OCR."]


def test_split_text_respects_limit_and_rejects_nonpositive_limit():
    chunks = split_text("Premier paragraphe assez long.\n\nSecond paragraphe.", 25)
    assert all(len(chunk) <= 25 for chunk in chunks)
    try:
        split_text("texte", 0)
    except ValueError as exc:
        assert "supérieur à zéro" in str(exc)
    else:
        raise AssertionError("Un découpage avec une limite nulle doit échouer.")


def test_docx_extraction_includes_table_content(tmp_path: Path):
    from docx import Document

    docx_path = tmp_path / "rapport.docx"
    document = Document()
    document.add_paragraph("Introduction du rapport.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Méthode"
    table.cell(0, 1).text = "TextRank"
    document.save(docx_path)

    extracted = extract_document(docx_path)

    assert extracted.pages is None
    assert "Introduction du rapport." in extracted.text
    assert "Méthode | TextRank" in extracted.text


class FakeToken:
    def __init__(self, text: str, lemma: str | None = None, *, is_stop: bool = False, like_num: bool = False):
        self.text = text
        self.lemma_ = lemma or text.casefold()
        self.is_space = False
        self.is_punct = not text.isalnum()
        self.like_num = like_num
        self.is_alpha = text.isalpha()
        self.is_stop = is_stop


class FakeSentence:
    def __init__(self, text: str, tokens: list[FakeToken]):
        self.text = text
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)


def test_preprocess_filters_short_and_numeric_sentences(monkeypatch):
    sentences = [
        FakeSentence("Trop court.", [FakeToken("Trop"), FakeToken("court"), FakeToken(".")]),
        FakeSentence("2020 2021 2022 2023 2024.", [
            FakeToken(str(number), like_num=True) for number in range(5)
        ]),
        FakeSentence("Les chercheurs analysent plusieurs données importantes.", [
            FakeToken("Les", is_stop=True),
            FakeToken("chercheurs", "chercheur"),
            FakeToken("analysent", "analyser"),
            FakeToken("plusieurs"),
            FakeToken("données", "donnée"),
            FakeToken("importantes", "important"),
            FakeToken("."),
        ]),
    ]

    class FakeNlp:
        max_length = 100

        def __call__(self, text):
            assert text == "Texte français"
            return SimpleNamespace(sents=sentences)

    monkeypatch.setattr(preprocessing, "_load_nlp", lambda: FakeNlp())
    processed = preprocessing.preprocess_text("Texte français")

    assert processed.sentences == ["Les chercheurs analysent plusieurs données importantes."]
    assert processed.normalized_sentences == ["chercheur analyser plusieurs donnée important"]


def test_extractive_methods_keep_source_sentences_and_lead3_order():
    sentences = [
        "Première phrase de contexte et de présentation.",
        "Deuxième phrase qui développe le sujet.",
        "Troisième phrase qui précise les enjeux.",
        "Quatrième phrase avec un résultat important.",
        "Cinquième phrase qui détaille la conclusion.",
    ]
    normalized = [
        "premier phrase contexte présentation",
        "deuxième phrase développer sujet",
        "troisième phrase préciser enjeu",
        "quatrième phrase résultat important",
        "cinquième phrase détailler conclusion",
    ]

    assert summarize_extractive("lead3", sentences, normalized) == " ".join(sentences[:3])
    for method in ("tfidf", "textrank", "lsa", "lexrank"):
        summary = summarize_extractive(method, sentences, normalized, 0.2)
        assert summary in sentences


def test_barthez_uses_sentence_chunks_and_final_consolidation(monkeypatch):
    from app import local_model

    calls = []

    def generate(source, instruction, max_new_tokens):
        calls.append((source, instruction, max_new_tokens))
        return "Résumé intermédiaire."

    monkeypatch.setattr(local_model, "summarize_with_barthez", generate)
    sentences = [f"Cette phrase française décrit le sujet numéro {index} en détail." for index in range(60)]

    summary, chunks = summarize_abstractive(
        " ".join(sentences),
        "court",
        model_name="barthez",
        sentences=sentences,
    )

    assert chunks > 1
    assert len(calls) == chunks + 1
    assert summary == "Résumé intermédiaire."
    assert all(len(source) <= 2700 for source, _, _ in calls[:chunks])


def test_barthez_generation_uses_local_seq2seq_model(monkeypatch):
    from app import local_model

    class FakeTokenizer:
        def __call__(self, source, **options):
            assert "sujet du document" in source
            assert options["max_length"] == 1024
            return {"input_ids": [1, 2, 3]}

        def decode(self, tokens, skip_special_tokens):
            assert tokens == [1, 2, 3]
            assert skip_special_tokens
            return "Résumé BARThez."

    class FakeModel:
        def generate(self, **options):
            assert options["max_new_tokens"] == 300
            assert options["num_beams"] == 4
            return [[1, 2, 3]]

    monkeypatch.setattr(local_model, "_load_barthez", lambda: (FakeTokenizer(), FakeModel()))

    assert local_model.summarize_with_barthez("Texte source.", "sujet du document") == "Résumé BARThez."


def test_pipeline_response_contains_plan_fields(monkeypatch):
    preprocessed = PreprocessedDocument(
        text="Texte nettoyé",
        sentences=["Une phrase de résumé exploitable."],
        normalized_sentences=["phrase résumé exploitable"],
        language="fr",
        warnings=["langue française"],
    )
    def fake_preprocess(text):
        assert text == "Un texte source suffisamment long pour compter plusieurs mots."
        return preprocessed

    def fake_summary(method, sentences, normalized_sentences, ratio):
        assert (method, len(sentences), len(normalized_sentences), ratio) == ("textrank", 1, 1, 0.2)
        return "Résumé."

    def fake_keywords(text, method, top_k):
        assert (text, method, top_k) == ("Texte nettoyé", "yake", 10)
        return [{"terme": "résumé", "score": 0.9}]

    monkeypatch.setattr(pipeline, "preprocess_text", fake_preprocess)
    monkeypatch.setattr(pipeline, "summarize_extractive", fake_summary)
    monkeypatch.setattr(pipeline, "extract_keywords", fake_keywords)

    result = pipeline.run_pipeline(
        "rapport.pdf",
        ExtractionResult("Un texte source suffisamment long pour compter plusieurs mots.", 2, ["OCR page 1"]),
    )

    assert result["filename"] == "rapport.pdf"
    assert result["fichier"] == "rapport.pdf"
    assert result["langue"] == "fr"
    assert result["methode_resume"] == "textrank"
    assert result["resume"] == result["summary"]
    assert result["mots_cles"] == [{"terme": "résumé", "score": 0.9}]
    assert result["texte_pretraite"] == "Texte nettoyé"
    assert result["phrases_analysees"] == 1
    assert result["avertissements"] == ["OCR page 1", "langue française"]
    assert result["duree_s"] >= 0


def test_keyword_metrics_use_lemma_sets(monkeypatch):
    monkeypatch.setattr(evaluation, "_lemmatize", lambda terms: {"système" if term.startswith("système") else term for term in terms})
    scores = evaluation.keyword_scores(["systèmes", "modèles"], ["système", "données"], 5)
    assert scores["precision_at_5"] == 0.5
    assert scores["recall_at_5"] == 0.5
    assert scores["f1_at_5"] == 0.5


def test_rouge_tokenizer_preserves_french_accents():
    scores = evaluation.summary_scores("été", "été")

    assert scores["rouge1_fmeasure"] == 1.0
    assert scores["rougeL_fmeasure"] == 1.0


def test_benchmark_runs_every_extractive_and_keyword_method(monkeypatch):
    preprocessed = PreprocessedDocument(
        text="Texte prétraité",
        sentences=["Une phrase de résumé exploitable."],
        normalized_sentences=["phrase résumé exploitable"],
        language="fr",
        warnings=[],
    )

    def fake_preprocess(text):
        assert text == "Texte source"
        return preprocessed

    def fake_summary(method, sentences, normalized_sentences, ratio):
        assert sentences == preprocessed.sentences
        assert normalized_sentences == preprocessed.normalized_sentences
        assert ratio == 0.2
        return method

    def fake_keywords(text, method, top_k):
        assert (text, top_k) == ("Texte prétraité", 10)
        return [{"terme": method, "score": 1.0}]

    def fake_summary_scores(prediction, reference, include_bertscore):
        assert isinstance(prediction, str)
        assert reference == "Référence"
        assert include_bertscore is False
        return {"rouge1_fmeasure": 0.5}

    def fake_keyword_scores(predicted, reference, k):
        assert predicted[0]["terme"] in {"yake", "keybert"}
        assert reference == ["test"]
        assert k in {5, 10}
        return {"f1": 0.5}

    monkeypatch.setattr(evaluation, "preprocess_text", fake_preprocess)
    monkeypatch.setattr(evaluation, "summarize_extractive", fake_summary)
    monkeypatch.setattr(evaluation, "extract_keywords", fake_keywords)
    monkeypatch.setattr(evaluation, "summary_scores", fake_summary_scores)
    monkeypatch.setattr(evaluation, "keyword_scores", fake_keyword_scores)

    results = evaluation.benchmark_records([{
        "fichier": "rapport.pdf",
        "texte": "Texte source",
        "resume_reference": "Référence",
        "mots_cles_reference": ["test"],
    }])

    detail_rows = [row for row in results if row["fichier"] == "rapport.pdf"]
    assert len(detail_rows) == 10
    assert {row["methode_resume"] for row in detail_rows} == {
        "lead3", "tfidf", "textrank", "lexrank", "lsa"
    }
    assert {row["methode_mots_cles"] for row in detail_rows} == {"yake", "keybert"}


def test_health_route_supports_short_and_api_paths():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    api_response = client.get("/api/health")
    short_response = client.get("/health")

    assert api_response.status_code == 200
    assert short_response.status_code == 200
    assert short_response.json() == api_response.json()


def test_summarize_api_returns_integrated_pipeline_fields(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    def fake_run_pipeline(filename, extracted, method, rate, keyword_method, top_k, length):
        assert (top_k, length) == (10, "court")
        return {
            "filename": filename,
            "summary": "Résumé de test.",
            "mots_cles": [{"terme": "test", "score": 0.9}],
            "methode_resume": method,
            "methode_mots_cles": keyword_method,
            "taux": rate / 100,
            "avertissements": extracted.warnings,
        }

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    client = TestClient(main.app)
    response = client.post(
        "/api/summarize",
        files={"file": ("rapport.txt", b"Texte exploitable " * 20, "text/plain")},
        data={"method": "textrank", "rate": "20", "keyword_method": "yake", "top_k": "10"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["methode_resume"] == "textrank"
    assert result["mots_cles"][0]["terme"] == "test"
    assert result["model"] == "TextRank"
    assert result["summary"] == "Résumé de test."


def test_docx_export_includes_summary_and_keywords():
    from docx import Document
    from fastapi.testclient import TestClient

    from app.main import app

    response = TestClient(app).post(
        "/api/export/docx",
        json={
            "filename": "rapport.pdf",
            "summary": "Résumé de test.",
            "pages": 2,
            "model": "TextRank",
            "keywords": [{"terme": "langage naturel", "score": 0.9}],
        },
    )

    assert response.status_code == 200
    exported = Document(BytesIO(response.content))
    exported_text = "\n".join(paragraph.text for paragraph in exported.paragraphs)
    assert "Résumé de test." in exported_text
    assert "langage naturel" in exported_text
