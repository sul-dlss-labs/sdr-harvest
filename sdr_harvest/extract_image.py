from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

from .core import (
    IMAGE_DESCRIPTION_MODEL,
    IMAGE_EXTRACT_SIGNATURE,
    StageError,
    TransientStageError,
)
from .download import read_derivatives
from .embed import http_error_message, retry_after_seconds
from .manifests import object_type


GEMINI_GENERATE_URL = (
    "https://dlss-aigateway-prod.stanford.edu/gemini/v1beta/models/"
    f"{IMAGE_DESCRIPTION_MODEL}:generateContent"
)

# Changing this prompt requires a new IMAGE_EXTRACT_SIGNATURE prompt version.
DESCRIPTION_PROMPT = """\
Describe this image from the Stanford Digital Repository so that people can \
find it with a text search.

Catalog title: {title}
Catalog subjects: {subjects}

Use the catalog title and subjects only to name things that are visible in the \
image, such as a named ship or place. Do not state anything from them that the \
image itself does not show.

Write one to three short paragraphs of plain prose, without a preamble or \
headings, covering:
- what is depicted: the setting, objects, and any people, with what they are \
doing and how many there are;
- the medium and format when evident, such as a black-and-white photographic print.

Do not name or guess the identity of any person, and do not guess anyone's age, \
race, ethnicity, or other personal characteristics. Do not speculate beyond what \
is visible.

If the image contains legible text, such as a caption, sign, or label, end with \
a line that starts with "Visible text:" followed by a verbatim transcription. \
Leave that line out if there is no legible text."""


def cocina_title(cocina: dict) -> str:
    for title in (cocina.get("description") or {}).get("title", []):
        if title.get("value"):
            return str(title["value"])
        parts = [
            str(part["value"])
            for part in title.get("structuredValue", [])
            if part.get("value")
        ]
        if parts:
            return " ".join(parts)
    return str(cocina.get("label") or "")


def cocina_subjects(cocina: dict) -> list[str]:
    subjects: list[str] = []
    for subject in (cocina.get("description") or {}).get("subject", []):
        if subject.get("value"):
            subjects.append(str(subject["value"]))
            continue
        parts = [
            str(part["value"])
            for part in subject.get("structuredValue", [])
            if part.get("value")
        ]
        if parts:
            subjects.append(" -- ".join(parts))
    return subjects


def description_prompt(context: dict) -> str:
    return DESCRIPTION_PROMPT.format(
        title=context["title"] or "(none)",
        subjects="; ".join(context["subjects"]) or "(none)",
    )


def describe_image(image: Path, prompt: str, key: str) -> str:
    """Ask the vision model for a searchable description of one image."""
    try:
        response = requests.post(
            GEMINI_GENERATE_URL,
            params={"key": key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64.b64encode(
                                        image.read_bytes()
                                    ).decode("ascii"),
                                }
                            },
                            {"text": prompt},
                        ],
                    }
                ],
            },
            timeout=180,
        )
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise TransientStageError(str(exc).replace(key, "[REDACTED]")) from exc
    if response.status_code == 429 or response.status_code >= 500:
        raise TransientStageError(
            http_error_message(response), retry_after=retry_after_seconds(response)
        )
    if response.status_code != 200:
        raise StageError(http_error_message(response))
    try:
        candidate = response.json()["candidates"][0]
        text = "".join(
            part.get("text", "")
            for part in (candidate.get("content") or {}).get("parts", [])
            if not part.get("thought")
        ).strip()
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise StageError(
            f"Image description response had no text for {image.name}: "
            f"{response.text[:500]}"
        ) from exc
    if not text:
        raise StageError(
            f"Image description was empty for {image.name} "
            f"(finishReason={candidate.get('finishReason')})"
        )
    return text


class ImageDescriptionStrategy:
    """Describe the IIIF derivatives of an image object with a vision model."""

    signature = IMAGE_EXTRACT_SIGNATURE

    def supports(self, cocina: dict, source_files: list[Path]) -> bool:
        return object_type(cocina) == "image" and any(
            path.suffix.lower() == ".jpg" for path in source_files
        )

    def context(self, cocina: dict) -> dict:
        """The descriptive metadata the prompt reads, so reuse can account for it."""
        return {"title": cocina_title(cocina), "subjects": cocina_subjects(cocina)}

    def extract(
        self,
        source_files: list[Path],
        output: Path,
        source_pages: dict[str, str] | None = None,
        cocina: dict | None = None,
    ) -> dict[str, dict[str, object]]:
        images = [path for path in source_files if path.suffix.lower() == ".jpg"]
        if not images:
            return {}
        key = os.environ.get("LITELLM_API_KEY")
        if not key:
            raise StageError("LITELLM_API_KEY is not set")
        derivatives = read_derivatives(images[0].parent)
        prompt = description_prompt(self.context(cocina or {}))
        extracted: dict[str, dict[str, object]] = {}
        for image in images:
            source_file = derivatives.get(image.name, {}).get("source_file")
            if not source_file:
                raise StageError(f"No source JP2 is recorded for {image.name}")
            target = output / f"{image.stem}.md"
            temporary = target.with_suffix(".md.tmp")
            temporary.write_text(describe_image(image, prompt, key), encoding="utf-8")
            temporary.replace(target)
            extracted[target.name] = {
                "page": (source_pages or {}).get(source_file),
                "source_file": source_file,
            }
        return extracted
