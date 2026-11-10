from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from .core import VTT_EXTRACT_SIGNATURE, StageError


TIMESTAMP = r"(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3}"
TIMING_LINE = re.compile(rf"^({TIMESTAMP})\s+-->\s+({TIMESTAMP})(?:\s.*)?$")
# Captures the annotation of a voice span: "<v Henry>" or the "<v ->" that
# caption vendors use to mark an unnamed change of speaker.
VOICE_SPAN = re.compile(r"<v(?:\.[^\s>]*)?(?:[ \t]+([^>]*))?>")
MARKUP = re.compile(r"<[^>]*>")
# Cues that only describe sound, such as "(silence)", "[laughs]", or "♪".
NON_SPEECH = re.compile(r"[\[(][^\])]*[\])]|[\s♪♫]+")
ANONYMOUS_SPEAKERS = {"", "-"}


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str
    # None continues the current speaker; "" marks an unnamed new speaker.
    speaker: str | None


def timestamp_seconds(value: str) -> float:
    seconds = 0.0
    for part in value.replace(",", ".").split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def parse_vtt(path: Path) -> list[Cue]:
    """Read the spoken text and timings of a WebVTT file's cues."""
    try:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise StageError(f"Unreadable caption file {path.name}: {exc}") from exc
    blocks = re.split(
        r"\n[ \t]*\n", raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    if not blocks[0].startswith("WEBVTT"):
        raise StageError(f"Caption file is not WebVTT: {path.name}")

    cues: list[Cue] = []
    pending_speaker: str | None = None
    for block in blocks[1:]:
        lines = block.split("\n")
        # A cue may begin with an identifier line; NOTE, STYLE, and REGION
        # blocks have no timing line and are skipped.
        timing_index = next(
            (index for index, line in enumerate(lines[:2]) if "-->" in line), None
        )
        if timing_index is None:
            continue
        timing = TIMING_LINE.match(lines[timing_index].strip())
        if timing is None:
            continue
        payload = " ".join(
            line.strip() for line in lines[timing_index + 1 :] if line.strip()
        )
        voice = VOICE_SPAN.search(payload)
        speaker = None
        if voice:
            speaker = (voice.group(1) or "").strip()
            speaker = "" if speaker in ANONYMOUS_SPEAKERS else speaker
        text = " ".join(html.unescape(MARKUP.sub("", payload)).split())
        if not text or NON_SPEECH.fullmatch(text):
            # Keep a speaker change carried by a dropped cue for the next one.
            pending_speaker = speaker if speaker is not None else pending_speaker
            continue
        if speaker is None:
            speaker, pending_speaker = pending_speaker, None
        start = timestamp_seconds(timing.group(1))
        end = max(start, timestamp_seconds(timing.group(2)))
        if cues and speaker is None and text == cues[-1].text:
            # Some captioning repeats a line across consecutive cues.
            cues[-1] = Cue(cues[-1].start, end, text, cues[-1].speaker)
            continue
        cues.append(Cue(start, end, text, speaker))
    return cues


def render_transcript(cues: list[Cue]) -> tuple[str, list[list[float]]]:
    """Render one cue per line, with a blank line before each new speaker.

    Returns the text and each cue's ``[char_start, char_end, start_s, end_s]``.
    """
    parts: list[str] = []
    spans: list[list[float]] = []
    offset = 0
    current_speaker: str | None = None
    for cue in cues:
        new_speaker = cue.speaker == "" or (
            cue.speaker is not None and cue.speaker != current_speaker
        )
        if cue.speaker is not None:
            current_speaker = cue.speaker
        line = cue.text
        if new_speaker and cue.speaker:
            line = f"{cue.speaker}: {cue.text}"
        separator = "" if not parts else "\n\n" if new_speaker else "\n"
        offset += len(separator)
        spans.append([offset, offset + len(line), cue.start, cue.end])
        parts.append(separator + line)
        offset += len(line)
    return "".join(parts), spans


class VttExtractionStrategy:
    """Extract spoken text and cue timings from WebVTT caption files."""

    signature = VTT_EXTRACT_SIGNATURE

    def supports(self, cocina: dict, source_files: list[Path]) -> bool:
        return any(path.suffix.lower() == ".vtt" for path in source_files)

    def extract(
        self,
        source_files: list[Path],
        output: Path,
        source_pages: dict[str, str] | None = None,
    ) -> dict[str, dict[str, object]]:
        extracted: dict[str, dict[str, object]] = {}
        for vtt in source_files:
            if vtt.suffix.lower() != ".vtt":
                continue
            target = output / f"{vtt.stem}.md"
            if target.name in extracted:
                raise StageError(
                    f"Multiple caption files map to the same Markdown file: "
                    f"{target.name}"
                )
            text, cues = render_transcript(parse_vtt(vtt))
            temporary = target.with_suffix(".md.tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(target)
            extracted[target.name] = {"cues": cues, "source_file": vtt.name}
        return extracted
