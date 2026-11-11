from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from .core import StageError, fingerprint


DRUID_RE = re.compile(r"^(?:druid:)?([a-z]{2}\d{3}[a-z]{2}\d{4})$", re.I)
# File access.view values whose content may be indexed. Only "world" is public.
INDEXABLE_VIEWS = {"world", "stanford"}
CAPTION_MIME_TYPE = "text/vtt"
MEDIA_MIME_PREFIXES = ("audio/", "video/")
# The files sul-embed plays for each resource type (its primary_mimetypes
# setting), so caption chunks name the file a viewer selects by filename.
PLAYABLE_MIME_TYPES = {
    "audio": {"audio/aac", "audio/mpeg", "audio/mp4"},
    "video": {"video/mp4", "video/quicktime"},
}


class MediaFile(NamedTuple):
    """The file a resource's player opens, and whether the resource has video."""

    filename: str
    is_video: bool
ENGLISH_LANGUAGE_TAGS = {"en", "eng"}


def read_druids(path: Path) -> set[str]:
    """Return the first DRUID-shaped value of each row in a CSV file."""
    druids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    for row in rows:
        for value in row:
            match = DRUID_RE.match(value.strip())
            if match:
                druids.add(match.group(1).lower())
                break
    return druids


def parse_manifest(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    druids = read_druids(path)
    if not druids:
        raise ValueError(f"No DRUIDs found in {path}")
    return druids


def merge_manifests(
    inputs: list[Path], output: Path, exclude: set[str] = frozenset()
) -> dict:
    """Merge manifest files into a deterministic, deduplicated DRUID CSV.

    DRUIDs in ``exclude`` (the dark list) are omitted from the output.
    """
    if len(inputs) < 2:
        raise ValueError("At least two input manifests are required")
    resolved_inputs = {path.resolve() for path in inputs}
    if output.resolve() in resolved_inputs:
        raise ValueError("Output manifest must not overwrite an input manifest")

    per_input = {str(path): len(parse_manifest(path)) for path in inputs}
    merged: set[str] = set()
    for path in inputs:
        merged.update(parse_manifest(path))
    excluded = merged & exclude
    merged -= excluded

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identifier"])
        writer.writerows((druid,) for druid in sorted(merged))
    temporary.replace(output)
    return {
        "inputs": per_input,
        "input_records": sum(per_input.values()),
        "unique_records": len(merged),
        "duplicates_removed": sum(per_input.values()) - len(merged) - len(excluded),
        "dark_excluded": len(excluded),
        "output": str(output),
    }


def _cocina_files(
    data: dict,
) -> tuple[list[dict], set[str], dict[str, MediaFile]]:
    """Collect indexable text files, page resources, and each resource's media."""
    found: list[dict] = []
    page_resources: set[str] = set()
    media_files: dict[str, list[tuple[str, str, str | None]]] = {}
    resource_number = 0
    page_number = 0

    def visit(
        node: object,
        resource_type: str | None = None,
        resource_key: str | None = None,
        resource_page: int | None = None,
    ) -> None:
        nonlocal page_number, resource_number
        if isinstance(node, dict):
            node_type = str(node.get("type", ""))
            if "/models/resources/" in node_type:
                resource_type = node_type.rsplit("/", 1)[-1]
                resource_number += 1
                resource_key = str(
                    node.get("externalIdentifier") or f"resource:{resource_number}"
                )
                if resource_type == "page":
                    page_number += 1
                    resource_page = page_number
                    page_resources.add(resource_key)
                else:
                    resource_page = None
            mime_type = str(node.get("hasMimeType", "")).lower()
            if (
                mime_type.startswith(MEDIA_MIME_PREFIXES)
                and node.get("filename")
                and resource_key is not None
            ):
                media_files.setdefault(resource_key, []).append(
                    (node["filename"], mime_type, resource_type)
                )
            is_pdf = mime_type == "application/pdf"
            is_transcription_xml = mime_type in {
                "application/xml",
                "text/xml",
            } and str(node.get("use", "")).lower() == "transcription"
            is_caption = mime_type == CAPTION_MIME_TYPE
            view = str((node.get("access") or {}).get("view", "")).lower()
            if (
                (is_pdf or is_transcription_xml or is_caption)
                and node.get("filename")
                and view in INDEXABLE_VIEWS
            ):
                digests = {
                    digest.get("type"): digest.get("digest")
                    for digest in node.get("hasMessageDigests", [])
                }
                found.append(
                    {
                        "file_id": str(
                            node.get("externalIdentifier") or node["filename"]
                        ),
                        "filename": node["filename"],
                        "size": node.get("size"),
                        "version": (
                            str(node.get("version"))
                            if node.get("version") is not None
                            else None
                        ),
                        "sha1": digests.get("sha1"),
                        "md5": digests.get("md5"),
                        "_resource_type": resource_type,
                        "_resource_key": resource_key,
                        "_resource_page": resource_page,
                        "_mime_type": mime_type,
                        "_use": str(node.get("use", "")).lower(),
                        "_language": str(node.get("languageTag") or "").lower(),
                    }
                )
            for value in node.values():
                visit(value, resource_type, resource_key, resource_page)
        elif isinstance(node, list):
            for value in node:
                visit(value, resource_type, resource_key, resource_page)

    visit(data.get("structural", {}))
    media_by_resource = {
        resource: MediaFile(
            next(
                (
                    filename
                    for filename, mime_type, resource_type in files
                    if mime_type in PLAYABLE_MIME_TYPES.get(resource_type or "", ())
                ),
                files[0][0],
            ),
            any(mime_type.startswith("video/") for _, mime_type, _ in files),
        )
        for resource, files in media_files.items()
    }
    return found, page_resources, media_by_resource


def _public_file(item: dict) -> dict:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _sorted_public_files(items: Iterable[dict]) -> list[dict]:
    return sorted(
        (_public_file(item) for item in items),
        key=lambda item: (item["filename"], item["file_id"]),
    )


def _caption_rank(item: dict) -> tuple[bool, str]:
    """Order a resource's caption tracks: English first, then by filename."""
    language = item["_language"].split("-", 1)[0]
    return (language not in ENGLISH_LANGUAGE_TAGS, item["filename"])


def _captions_by_media_resource(
    found: list[dict], media_by_resource: dict[str, MediaFile]
) -> dict[str, dict]:
    """Choose one indexable caption track for each resource with media."""
    tracks: dict[str, list[dict]] = {}
    for item in found:
        resource = item["_resource_key"]
        if item["_mime_type"] == CAPTION_MIME_TYPE and resource in media_by_resource:
            tracks.setdefault(resource, []).append(item)
    return {
        resource: min(items, key=_caption_rank) for resource, items in tracks.items()
    }


def cocina_pdf_files(data: dict) -> list[dict]:
    found = [
        item
        for item in _cocina_files(data)[0]
        if item["_mime_type"] == "application/pdf"
    ]
    if any(item["_resource_type"] == "object" for item in found):
        found = [item for item in found if item["_resource_type"] != "page"]
    return _sorted_public_files(found)


def cocina_source_files(data: dict) -> list[dict]:
    """Select the text sources to index for an object.

    Books with complete page-level ALTO use the ALTO. Otherwise, when every
    audio or video file has an indexable WebVTT caption track, the captions
    are used (one track per media file, English preferred) because their cue
    timings can link into the media. Captioned videos also count as complete
    when the only uncaptioned media are no more audio files than there are
    captioned videos, the audio-only copies that often accompany recorded
    interviews. Otherwise partial captions are used only when the object has
    no PDF to fall back on.
    """
    found, page_resources, media_by_resource = _cocina_files(data)
    is_book = str(data.get("type", "")).rsplit("/", 1)[-1] == "book"
    alto_by_page: dict[str | None, list[dict]] = {}
    for item in found:
        if (
            item["_resource_type"] == "page"
            and item["_mime_type"] in {"application/xml", "text/xml"}
            and item["_use"] == "transcription"
        ):
            alto_by_page.setdefault(item["_resource_key"], []).append(item)
    has_complete_alto = bool(page_resources) and all(
        len(alto_by_page.get(resource, [])) == 1 for resource in page_resources
    )
    if is_book and has_complete_alto:
        alto = [alto_by_page[resource][0] for resource in page_resources]
        return _sorted_public_files(alto)

    pdfs = cocina_pdf_files(data)
    captions = _captions_by_media_resource(found, media_by_resource)
    uncaptioned = media_by_resource.keys() - captions.keys()
    captioned_videos = sum(media_by_resource[key].is_video for key in captions)
    # Uncaptioned audio may be copies of captioned videos, at most one each.
    uncaptioned_are_audio_copies = len(uncaptioned) <= captioned_videos and not any(
        media_by_resource[key].is_video for key in uncaptioned
    )
    has_complete_captions = bool(captions) and uncaptioned_are_audio_copies
    if has_complete_captions or (captions and not pdfs):
        return _sorted_public_files(captions.values())
    return pdfs


def cocina_page_numbers(data: dict) -> dict[str, str]:
    """Map files in page resources to their 1-based structural page number."""
    found, _, _ = _cocina_files(data)
    return {
        item["filename"]: str(item["_resource_page"])
        for item in found
        if item["_resource_page"] is not None
    }


def cocina_caption_media_files(data: dict) -> dict[str, str]:
    """Map each indexable caption file to the media file in its resource."""
    found, _, media_by_resource = _cocina_files(data)
    return {
        item["filename"]: media_by_resource[item["_resource_key"]].filename
        for item in found
        if item["_mime_type"] == CAPTION_MIME_TYPE
        and item["_resource_key"] in media_by_resource
    }


def cocina_public_files(data: dict) -> set[str]:
    """Return the names of files whose COCINA access.view is world."""
    public: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if (
                node.get("filename")
                and str((node.get("access") or {}).get("view", "")).lower()
                == "world"
            ):
                public.add(node["filename"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(data.get("structural", {}))
    return public


def source_fingerprint(data: dict, files: list[dict]) -> str:
    # The whole record matters because metadata is embedded along with PDF text.
    return fingerprint({"cocina": data, "pdfs": files})


def safe_name(filename: str) -> str:
    """Return a traversal-safe artifact filename."""
    name = Path(filename).name
    if name in {"", ".", ".."} or name != filename:
        raise StageError(f"Unsafe filename: {filename!r}")
    return name
