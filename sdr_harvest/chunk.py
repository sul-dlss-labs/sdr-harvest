from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .manifests import cocina_page_numbers, cocina_public_files


CHUNK_SIZE = 1_500
CHUNK_OVERLAP = 200
# Joins consecutive page files. A line break rather than a paragraph break, so
# the splitter treats the page break like any other line and chunks overlap
# across it.
PAGE_SEPARATOR = "\n"


def page_spans(page_ranges: list[dict], start: int, end: int) -> list[dict]:
    """Return the non-empty page ranges that overlap ``[start, end)``."""
    return [
        page_range
        for page_range in page_ranges
        if page_range["start"] < page_range["end"]
        and page_range["start"] < end
        and start < page_range["end"]
    ]


def chunk_pages(
    attributes: dict, start: int, end: int
) -> tuple[list[str] | None, list[int] | None]:
    """Return the pages a chunk spans and where each starts in the chunk text.

    The first page is the one the chunk starts on. Offsets are only returned
    when the chunk spans more than one page.
    """
    spans = page_spans(attributes.get("pages", []), start, end)
    if not spans:
        page = attributes.get("page")
        return ([str(page)] if page is not None else None), None
    pages = [str(page_range["page"]) for page_range in spans]
    if len(pages) == 1:
        return pages, None
    return pages, [max(page_range["start"] - start, 0) for page_range in spans]


def page_sequences(
    markdown_files: list[Path], page_metadata: dict, cocina: dict
) -> list[list[tuple[Path, int | None]]]:
    """Group Markdown files into the units that are each split into chunks.

    A file holding a single page of the object's structural sequence (an ALTO
    page or a one-page PDF) is joined with the files for the pages around it,
    so chunks overlap across page breaks as they do inside a multi-page PDF. A
    sequence ends where a page number is skipped, because that page has no
    indexed text, and where file access changes, so no chunk mixes public and
    restricted text. Every other file is a unit of its own.
    """
    structural = cocina_page_numbers(cocina)
    public = cocina_public_files(cocina)

    def page_number(markdown: Path) -> int | None:
        attributes = page_metadata.get(markdown.name, {})
        if len(attributes.get("pages", [])) > 1:
            return None
        page = structural.get(attributes.get("source_file"))
        return int(page) if page is not None else None

    numbered = [(markdown, page_number(markdown)) for markdown in markdown_files]
    units: list[list[tuple[Path, int | None]]] = [
        [(markdown, None)] for markdown, page in numbered if page is None
    ]
    previous: tuple[int, bool] | None = None
    for markdown, page in sorted(
        ((markdown, page) for markdown, page in numbered if page is not None),
        key=lambda item: (item[1], item[0].name),
    ):
        source_file = page_metadata[markdown.name].get("source_file")
        current = (page, source_file in public)
        if previous == (page - 1, current[1]):
            units[-1].append((markdown, page))
        else:
            units.append([(markdown, page)])
        previous = current
    return sorted(units, key=lambda unit: unit[0][0].name)


class Chunker:
    """Split extracted text and searchable metadata into Parquet rows."""

    def run(self, druid: str, version_dir: Path) -> Path:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
            add_start_index=True,
        )
        rows: list[dict] = []
        metadata = json.loads((version_dir / "metadata.json").read_text())
        metadata_text = []
        for key, value in metadata.items():
            if key in {"cocina_ss", "all_search_tesi"}:
                continue
            display = re.sub(r"_(tesi|ssim|isim|ss)$", "", key).replace("_", " ")
            rendered = ", ".join(map(str, value)) if isinstance(value, list) else value
            metadata_text.append(f"{display}: {rendered}")
        rows.append(
            {
                "object_id": druid,
                "file": "_metadata_",
                "source_file": "_metadata_",
                "pages": None,
                "page_start_offsets": None,
                "chunk_index": 0,
                "text": "\n".join(metadata_text),
            }
        )
        markdown_dir = version_dir / "markdown"
        page_manifest = markdown_dir / "pages.json"
        page_metadata = (
            json.loads(page_manifest.read_text(encoding="utf-8"))
            if page_manifest.exists()
            else {}
        )
        cocina_path = version_dir / "cocina.json"
        cocina = json.loads(cocina_path.read_text()) if cocina_path.exists() else {}
        for unit in page_sequences(
            sorted(markdown_dir.glob("*.md")), page_metadata, cocina
        ):
            if len(unit) == 1:
                rows.extend(
                    self._file_rows(druid, unit[0][0], page_metadata, splitter)
                )
            else:
                rows.extend(
                    self._sequence_rows(druid, unit, page_metadata, splitter)
                )
        output = version_dir / "chunks.parquet"
        schema = pa.schema(
            [
                ("object_id", pa.string()),
                ("file", pa.string()),
                ("source_file", pa.string()),
                ("pages", pa.list_(pa.string())),
                ("page_start_offsets", pa.list_(pa.int32())),
                ("chunk_index", pa.int32()),
                ("text", pa.string()),
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), output, compression="zstd"
        )
        return output

    def _file_rows(
        self,
        druid: str,
        markdown: Path,
        page_metadata: dict,
        splitter: RecursiveCharacterTextSplitter,
    ) -> list[dict]:
        attributes = page_metadata.get(markdown.name, {})
        content = markdown.read_text(encoding="utf-8")
        rows = []
        for index, chunk in enumerate(splitter.create_documents([content])):
            start = chunk.metadata["start_index"]
            pages, offsets = chunk_pages(
                attributes, start, start + len(chunk.page_content)
            )
            rows.append(
                {
                    "object_id": druid,
                    "file": markdown.name,
                    "source_file": attributes.get(
                        "source_file", markdown.with_suffix(".pdf").name
                    ),
                    "pages": pages,
                    "page_start_offsets": offsets,
                    "chunk_index": index,
                    "text": chunk.page_content,
                }
            )
        return rows

    def _sequence_rows(
        self,
        druid: str,
        unit: list[tuple[Path, int | None]],
        page_metadata: dict,
        splitter: RecursiveCharacterTextSplitter,
    ) -> list[dict]:
        """Chunk consecutive page files as one text.

        Each chunk is credited to the file it starts in and numbered among that
        file's chunks, which keeps child document IDs unique.
        """
        parts: list[str] = []
        page_ranges: list[dict] = []
        length = 0
        for markdown, page in unit:
            content = markdown.read_text(encoding="utf-8")
            if not content:
                continue
            if parts and not parts[-1][-1].isspace() and not content[0].isspace():
                parts.append(PAGE_SEPARATOR)
                length += len(PAGE_SEPARATOR)
            page_ranges.append(
                {
                    "page": str(page),
                    "start": length,
                    "end": length + len(content),
                    "file": markdown.name,
                    "source_file": page_metadata[markdown.name]["source_file"],
                }
            )
            parts.append(content)
            length += len(content)
        rows = []
        indexes: Counter[str] = Counter()
        for chunk in splitter.create_documents(["".join(parts)]):
            start = chunk.metadata["start_index"]
            end = start + len(chunk.page_content)
            first = page_spans(page_ranges, start, end)[0]
            pages, offsets = chunk_pages({"pages": page_ranges}, start, end)
            rows.append(
                {
                    "object_id": druid,
                    "file": first["file"],
                    "source_file": first["source_file"],
                    "pages": pages,
                    "page_start_offsets": offsets,
                    "chunk_index": indexes[first["file"]],
                    "text": chunk.page_content,
                }
            )
            indexes[first["file"]] += 1
        return rows
