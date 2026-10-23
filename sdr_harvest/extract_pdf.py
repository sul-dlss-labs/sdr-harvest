from __future__ import annotations

from concurrent.futures import Executor
from pathlib import Path

import pymupdf
import pymupdf4llm

from .core import PDF_EXTRACT_SIGNATURE, StageError


def _to_markdown(pdf: Path, pages: list[int] | None = None) -> list:
    result = pymupdf4llm.to_markdown(
        str(pdf),
        pages=pages,
        write_images=False,
        use_ocr=pymupdf4llm.ocr.OCRMode.NEVER,
        page_chunks=True,
    )
    return result if isinstance(result, list) else [{"text": str(result)}]


def _to_markdown_by_page(pdf: Path) -> list:
    """Convert one page at a time, leaving pages pymupdf4llm cannot handle empty.

    Works around https://github.com/pymupdf/pymupdf4llm/issues/495, where a
    single page raises and would otherwise fail the whole document.
    """
    with pymupdf.open(pdf) as document:
        page_count = document.page_count
    pages: list = []
    failures = 0
    for number in range(page_count):
        try:
            pages.extend(_to_markdown(pdf, [number]))
        except Exception:
            failures += 1
            pages.append({"text": "", "metadata": {"page_number": number + 1}})
    if failures == page_count:
        raise StageError(f"pymupdf4llm failed on every page of {pdf.name}")
    return pages


def extract_pdf_to_markdown(
    pdf: Path, target: Path, source_page: str | None = None
) -> dict[str, dict[str, object]]:
    """Extract one PDF in a process-pool-safe operation."""
    try:
        pages = _to_markdown(pdf)
    except Exception:
        pages = _to_markdown_by_page(pdf)
    page_texts: list[str] = []
    page_ranges: list[dict[str, str | int]] = []
    offset = 0
    for index, item in enumerate(pages, start=1):
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            metadata = item.get("metadata", {})
            internal_page = str(metadata.get("page_number", index))
        else:
            text = str(item)
            internal_page = str(index)
        page_texts.append(text)
        page_ranges.append(
            {
                "page": (
                    source_page if source_page and len(pages) == 1 else internal_page
                ),
                "start": offset,
                "end": offset + len(text),
            }
        )
        offset += len(text)
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text("".join(page_texts), encoding="utf-8")
    temporary.replace(target)
    return {
        target.name: {
            "pages": page_ranges,
            "source_file": pdf.name,
        }
    }


class PdfExtractionStrategy:
    """Extract embedded text from PDF source files."""

    signature = PDF_EXTRACT_SIGNATURE

    def __init__(self, executor: Executor | None = None) -> None:
        self.executor = executor

    def supports(self, cocina: dict, source_files: list[Path]) -> bool:
        return any(path.suffix.lower() == ".pdf" for path in source_files)

    def extract(
        self,
        source_files: list[Path],
        output: Path,
        source_pages: dict[str, str] | None = None,
    ) -> dict[str, dict[str, object]]:
        extracted: dict[str, dict[str, object]] = {}
        targets: set[str] = set()
        jobs = []
        for pdf in source_files:
            if pdf.suffix.lower() != ".pdf":
                continue
            target = output / f"{pdf.stem}.md"
            if target.name in targets:
                raise StageError(
                    f"Multiple PDFs map to the same Markdown file: {target.name}"
                )
            targets.add(target.name)
            source_page = (source_pages or {}).get(pdf.name)
            if self.executor:
                jobs.append(
                    self.executor.submit(
                        extract_pdf_to_markdown, pdf, target, source_page
                    )
                )
            else:
                extracted.update(extract_pdf_to_markdown(pdf, target, source_page))
        for job in jobs:
            extracted.update(job.result())
        return extracted
