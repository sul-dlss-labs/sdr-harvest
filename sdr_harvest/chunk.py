from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1_500
CHUNK_OVERLAP = 200


def chunk_pages(
    attributes: dict, start: int, end: int
) -> tuple[list[str] | None, list[int] | None]:
    """Return the pages a chunk spans and where each starts in the chunk text.

    The first page is the one the chunk starts on. Offsets are only returned
    when the chunk spans more than one page.
    """
    spans = [
        page_range
        for page_range in attributes.get("pages", [])
        if page_range["start"] < page_range["end"]
        and page_range["start"] < end
        and start < page_range["end"]
    ]
    if not spans:
        page = attributes.get("page")
        return ([str(page)] if page is not None else None), None
    pages = [str(page_range["page"]) for page_range in spans]
    if len(pages) == 1:
        return pages, None
    return pages, [max(page_range["start"] - start, 0) for page_range in spans]


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
        for markdown in sorted(markdown_dir.glob("*.md")):
            attributes = page_metadata.get(markdown.name, {})
            content = markdown.read_text(encoding="utf-8")
            chunks = splitter.create_documents([content])
            for index, chunk in enumerate(chunks):
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
