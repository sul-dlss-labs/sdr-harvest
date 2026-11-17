from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


ALTO_EXTRACT_SIGNATURE = "alto-xml-source-file-v2"
PDF_EXTRACT_SIGNATURE = "pymupdf4llm-1.28.2-no-ocr-v2"


class StageState(Protocol):
    status: str
    input_fingerprint: str | None
    stage_signature: str | None


@dataclass(frozen=True)
class StageDefinition:
    """A pipeline stage and the policy used to decide whether it is current."""

    name: str
    default_signature: str
    dynamic_signature: bool = False

    def signature(self, dynamic: str | None = None) -> str:
        if self.dynamic_signature and dynamic is not None:
            return dynamic
        return self.default_signature

    def is_current(
        self,
        record: StageState | None,
        input_fingerprint: str | None,
        signature: str,
        *,
        artifact_is_current: bool,
    ) -> bool:
        return bool(
            record
            and record.status == "succeeded"
            and record.input_fingerprint == input_fingerprint
            and record.stage_signature == signature
            and artifact_is_current
        )


STAGE_REGISTRY = (
    StageDefinition("cocina", "cocina-v2-conditional-get"),
    StageDefinition("download", "download-v1-sha1"),
    StageDefinition("metadata", "traject-sdr-config-v1"),
    StageDefinition("extract", PDF_EXTRACT_SIGNATURE, dynamic_signature=True),
    StageDefinition("chunk", "recursive-1500-200-v3"),
    StageDefinition("embed", "gemini-embedding-2-768-retrieval-document-v2"),
    StageDefinition("document", "nested-solr-document-v4-pages-every-chunk"),
)

STAGES = tuple(stage.name for stage in STAGE_REGISTRY)
BUILD_STAGES = STAGE_REGISTRY[1:]
SIGNATURES = {stage.name: stage.default_signature for stage in STAGE_REGISTRY}
