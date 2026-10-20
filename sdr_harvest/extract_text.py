from __future__ import annotations

import json
import shutil
from concurrent.futures import Executor
from pathlib import Path
from typing import Protocol

from .core import (
    StageError,
    directory_fingerprint,
    fingerprint,
    previous_versions,
)
from .extract_alto import AltoXmlExtractionStrategy
from .extract_pdf import PdfExtractionStrategy
from .manifests import cocina_page_numbers


# Records what produced a Markdown directory so later versions can reuse it.
REUSE_MARKER = "extract.json"


class ExtractionStrategy(Protocol):
    """An extraction implementation selected from object and source traits."""

    signature: str

    def supports(self, cocina: dict, source_files: list[Path]) -> bool: ...

    def extract(
        self,
        source_files: list[Path],
        output: Path,
        source_pages: dict[str, str] | None = None,
    ) -> dict[str, dict[str, object]]: ...


class TextExtractor:
    """Select a text extraction strategy and produce Markdown artifacts."""

    def __init__(
        self,
        strategies: list[ExtractionStrategy] | None = None,
        *,
        pdf_executor: Executor | None = None,
    ) -> None:
        self.strategies = strategies or [
            AltoXmlExtractionStrategy(),
            PdfExtractionStrategy(pdf_executor),
        ]

    def _strategy(
        self, version_dir: Path
    ) -> tuple[ExtractionStrategy, list[Path], dict]:
        source = version_dir / "pdfs"
        cocina = json.loads((version_dir / "cocina.json").read_text())
        source_files = sorted(path for path in source.iterdir() if path.is_file())
        strategy = next(
            (
                candidate
                for candidate in self.strategies
                if candidate.supports(cocina, source_files)
            ),
            None,
        )
        if strategy is None:
            raise StageError("No text extraction strategy supports this object")
        return strategy, source_files, cocina

    def signature(self, version_dir: Path) -> str:
        strategy, _, _ = self._strategy(version_dir)
        return strategy.signature

    def _provenance(self, version_dir: Path, files_fp: str) -> dict[str, str]:
        """Describe everything extraction reads; descriptive metadata is excluded."""
        strategy, _, cocina = self._strategy(version_dir)
        return {
            "signature": strategy.signature,
            "inputs": fingerprint(
                {"files": files_fp, "pages": cocina_page_numbers(cocina)}
            ),
        }

    def reusable_output(
        self,
        version_dir: Path,
        files_fp: str,
        legacy_output: Path | None = None,
    ) -> Path | None:
        """Find Markdown from an earlier version built from identical inputs.

        ``legacy_output`` is a directory the state store recorded as a current
        extraction with this signature; it predates provenance markers, so its
        inputs are recomputed from its own version directory.
        """
        wanted = self._provenance(version_dir, files_fp)
        for previous in previous_versions(version_dir):
            output = previous / "markdown"
            marker = output / REUSE_MARKER
            try:
                if marker.exists():
                    if json.loads(marker.read_text()) == wanted:
                        return output
                elif (
                    legacy_output is not None
                    and output.resolve() == legacy_output.resolve()
                    and (output / "pages.json").exists()
                    and self._provenance(
                        previous, directory_fingerprint(previous / "pdfs")
                    )
                    == wanted
                ):
                    return output
            except (OSError, ValueError, StageError):
                continue
        return None

    def run(
        self,
        version_dir: Path,
        files_fp: str | None = None,
        *,
        reuse_from: Path | None = None,
    ) -> Path:
        output = version_dir / "markdown"
        if files_fp is None:
            files_fp = directory_fingerprint(version_dir / "pdfs")
        provenance = self._provenance(version_dir, files_fp)
        if reuse_from is not None:
            shutil.rmtree(output, ignore_errors=True)
            shutil.copytree(reuse_from, output)
        else:
            output.mkdir(exist_ok=True)
            (output / REUSE_MARKER).unlink(missing_ok=True)
            strategy, source_files, cocina = self._strategy(version_dir)
            extracted = strategy.extract(
                source_files, output, cocina_page_numbers(cocina)
            )
            for stale in output.glob("*.md"):
                if stale.name not in extracted:
                    stale.unlink()
            manifest = output / "pages.json"
            temporary = manifest.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(extracted, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(manifest)
        # Written last so an interrupted extraction is never offered for reuse.
        (output / REUSE_MARKER).write_text(
            json.dumps(provenance, sort_keys=True), encoding="utf-8"
        )
        return output
