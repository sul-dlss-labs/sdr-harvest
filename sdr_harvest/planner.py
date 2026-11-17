from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from .cocina import cocina_checked_recently
from .core import SOURCE_INVENTORY_SIGNATURE, Settings, StageError
from .extract_text import TextExtractor
from .stages import BUILD_STAGES, STAGE_REGISTRY, StageDefinition
from .state import ObjectRecord, StateStore


@dataclass(frozen=True)
class WorkPlan:
    counts: dict[str, int]
    pending_druids: list[str]


def artifact_belongs_to_version(
    store: StateStore, artifact_path: str | None, version_dir: Path
) -> bool:
    """Return whether an existing recorded artifact belongs to a version."""
    if not artifact_path:
        return False
    path = store.resolve_path(artifact_path)
    return path.exists() and path.resolve().is_relative_to(version_dir.resolve())


class PipelinePlanner:
    """Build a conservative work plan from persisted pipeline state."""

    def __init__(
        self,
        settings: Settings,
        store: StateStore,
        *,
        extractor_factory: Callable[[], TextExtractor] = TextExtractor,
    ) -> None:
        self.settings = settings
        self.store = store
        self.extractor_factory = extractor_factory

    def plan(
        self, druids: list[str], *, show_progress: bool = True
    ) -> WorkPlan:
        counts = {stage.name: 0 for stage in STAGE_REGISTRY}
        pending: list[str] = []
        objects, stages = self.store.planning_snapshot(set(druids))
        extractor = self.extractor_factory()

        for druid in tqdm(
            druids,
            desc="Estimating remaining work",
            unit="object",
            disable=not show_progress,
        ):
            obj = objects.get(druid)
            input_fp = obj.source_fingerprint if obj else None
            needs_cocina = self._needs_cocina(druid, obj)
            needs_work = needs_cocina
            if needs_cocina:
                counts["cocina"] += 1

            dirty = not input_fp
            source_dir = self.settings.state_dir / "versions" / druid / str(input_fp)
            for stage in BUILD_STAGES:
                record = stages.get((druid, stage.name))
                signature = self._signature(stage, obj, extractor)
                artifact_is_current = bool(
                    record
                    and artifact_belongs_to_version(
                        self.store, record.artifact_path, source_dir
                    )
                )
                current = not dirty and stage.is_current(
                    record,
                    input_fp,
                    signature,
                    artifact_is_current=artifact_is_current,
                )
                if current:
                    input_fp = record.output_fingerprint or input_fp
                else:
                    counts[stage.name] += 1
                    dirty = True
                    needs_work = True
            if needs_work:
                pending.append(druid)

        return WorkPlan(counts, pending)

    def _needs_cocina(self, druid: str, obj: ObjectRecord | None) -> bool:
        cache_path = self.settings.state_dir / "sources" / druid / "cocina.json"
        return bool(
            not obj
            or not obj.source_cache_sha256
            or not cache_path.exists()
            or not cocina_checked_recently(obj.source_checked_at)
            or obj.source_inventory_signature != SOURCE_INVENTORY_SIGNATURE
        )

    def _signature(
        self,
        stage: StageDefinition,
        obj: ObjectRecord | None,
        extractor: TextExtractor,
    ) -> str:
        if not stage.dynamic_signature or not obj or not obj.current_artifact_dir:
            return stage.signature()
        version_dir = self.store.resolve_path(obj.current_artifact_dir)
        try:
            return stage.signature(extractor.signature(version_dir))
        except (OSError, ValueError, StageError):
            # Missing or invalid artifacts must be rebuilt conservatively.
            return stage.signature()
