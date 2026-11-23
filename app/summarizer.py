from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    pages: int | None
    warnings: list[str]


def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.casefold()).strip()


def _remove_repeated_margins(pages: list[list[str]]) -> list[list[str]]:
    if len(pages) < 3:
        return pages
    counts: dict[str, int] = {}
    for lines in pages:
        candidates = set(lines[:2] + lines[-2:])
        for line in candidates:
            key = _line_key(line)
            if key:
                counts[key] = counts.get(key, 0) + 1
    repeated = {
        key for key, count in counts.items()
        if count > len(pages) / 2
    }
    return [
        [line for line in lines if _line_key(line) not in repeated]
        for lines in pages
    ]


def _ordered_page_blocks(blocks: list[tuple], page_width: float) -> list[str]:
    text_blocks = [
        block for block in blocks
        if len(block) >= 7 and block[6] == 0 and str(block[4]).strip()
    ]
    midpoint = page_width / 2
    text_blocks.sort(key=lambda block: (block[0] >= midpoint, block[1], block[0]))
    return [str(block[4]).strip() for block in text_blocks]


def _extract_pdf(path: Path, force_ocr: bool) -> ExtractionResult:
    import pymupdf

    document = pymupdf.open(path)
    try:
        if document.needs_pass:
            raise ValueError("Le PDF est protégé par mot de passe.")
        page_lines: list[list[str]] = []
        warnings: list[str] = []
        for page_number, page in enumerate(document, start=1):
            blocks = page.get_text("blocks")
            page_text = "\n".join(_ordered_page_blocks(blocks, page.rect.width)).strip()
            if force_ocr or len(page_text) < 50:
                if not shutil.which("tesseract"):
                    raise RuntimeError(
                        "Tesseract est requis pour l'OCR. Installez Tesseract avec le paquet de langue française."
                    )
                import pytesseract
                from PIL import Image

                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(300 / 72, 300 / 72),
                    alpha=False,
                )
                with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
                    ocr_text = pytesseract.image_to_string(image, lang="fra").strip()
                if ocr_text:
                    page_text = ocr_text
                warnings.append(f"Page {page_number} traitée par OCR.")
            page_lines.append([line.strip() for line in page_text.splitlines() if line.strip()])

        page_lines = _remove_repeated_margins(page_lines)
        text = "\n\n".join("\n".join(lines) for lines in page_lines if lines).strip()
        return ExtractionResult(text, len(document), warnings)
    finally:
        document.close()


def extract_document(path: str | Path, force_ocr: bool = False) -> ExtractionResult:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, force_ocr)
    if force_ocr:
        raise ValueError("L'option OCR ne s'applique qu'aux fichiers PDF.")
    if suffix == ".docx":
        from docx import Document

        document = Document(path)
        blocks = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    blocks.append(" | ".join(cells))
        return ExtractionResult("\n\n".join(blocks), None, [])
    if suffix in {".txt", ".md"}:
        return ExtractionResult(path.read_text(encoding="utf-8", errors="replace"), None, [])
    raise ValueError("Format non pris en charge.")


def extract_text(path: str | Path) -> tuple[str, int | None]:
    """Compatibility wrapper for callers that only need extracted text and page count."""
    result = extract_document(path)
    return result.text, result.pages


def split_text(text: str, max_chars: int) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars doit être supérieur à zéro.")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            cut = paragraph.rfind(" ", 0, max_chars)
            cut = cut if cut >= max_chars // 2 else max_chars
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks
