import hashlib
import json
from pathlib import Path

DRUID = "ab123cd4567"


def cocina(version=1, digest="abc"):
    return {
        "externalIdentifier": f"druid:{DRUID}",
        "version": version,
        "description": {"title": [{"value": "Example"}]},
        "structural": {
            "contains": [
                {
                    "structural": {
                        "contains": [
                            {
                                "externalIdentifier": "file:1",
                                "filename": "example.pdf",
                                "hasMimeType": "application/pdf",
                                "access": {"view": "world"},
                                "size": 3,
                                "version": 1,
                                "hasMessageDigests": [
                                    {"type": "sha1", "digest": digest}
                                ],
                            }
                        ]
                    }
                }
            ]
        },
    }


def file_info(content: bytes) -> dict:
    return {
        "filename": "example.pdf",
        "size": len(content),
        "sha1": hashlib.sha1(content).hexdigest(),
        "md5": hashlib.md5(content).hexdigest(),
    }


class StreamingResponse:
    def __init__(
        self, content: bytes, status_code: int = 200, headers: dict | None = None
    ):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


def paged_cocina(*files):
    """Return COCINA with one page resource per (filename, view) pair."""
    return {
        "type": "https://cocina.sul.stanford.edu/models/book",
        "structural": {
            "contains": [
                {
                    "type": "https://cocina.sul.stanford.edu/models/resources/page",
                    "structural": {
                        "contains": [
                            {
                                "filename": filename,
                                "hasMimeType": (
                                    "application/pdf"
                                    if filename.endswith(".pdf")
                                    else "application/xml"
                                ),
                                "use": "transcription",
                                "access": {"view": view},
                            }
                        ]
                    },
                }
                for filename, view in files
            ]
        },
    }


def write_page_files(version_dir, cocina_data, pages):
    """Write extracted page Markdown as the ALTO or PDF strategy would."""
    version_dir.joinpath("metadata.json").write_text("{}")
    version_dir.joinpath("cocina.json").write_text(json.dumps(cocina_data))
    markdown = version_dir / "markdown"
    markdown.mkdir()
    manifest = {}
    for source_file, (page, text) in pages.items():
        name = Path(source_file).with_suffix(".md").name
        markdown.joinpath(name).write_text(text)
        manifest[name] = (
            {
                "pages": [{"page": page, "start": 0, "end": len(text)}],
                "source_file": source_file,
            }
            if source_file.endswith(".pdf")
            else {"page": page, "source_file": source_file}
        )
    markdown.joinpath("pages.json").write_text(json.dumps(manifest))
