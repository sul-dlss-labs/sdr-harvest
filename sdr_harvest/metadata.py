from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import requests

from .attempts import StageAttempts
from .cocina import (
    COCINA_MAX_AGE,
    CocinaCache,
    CocinaRefreshMachine,
    CocinaRefreshResult,
    CocinaRefreshState,
)
from .core import (
    SOURCE_INVENTORY_SIGNATURE,
    EventLog,
    Settings,
    StageError,
    TransientStageError,
)
from .stages import SIGNATURES
from .state import StateStore


class MetadataFetcher:
    """Fetch canonical COCINA and derive searchable metadata with Traject."""

    def __init__(
        self,
        settings: Settings,
        store: StateStore,
        http: requests.Session,
        attempts: StageAttempts,
    ) -> None:
        self.settings = settings
        self.store = store
        self.http = http
        self.attempts = attempts

    def fetch_cocina(
        self, run_id: int, druid: str
    ) -> tuple[Path, list[dict], str]:
        cache_dir = self.settings.state_dir / "sources" / druid
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "cocina.json"
        cache = CocinaCache.inspect(path, self.store.object_row(druid))
        machine = CocinaRefreshMachine(druid, cache, self.http)
        source_log = EventLog(
            self.settings.state_dir / "logs" / str(run_id) / druid / "cocina.jsonl"
        )

        if result := machine.use_fresh_cache():
            self._record_fresh_cache(run_id, druid, cache, result, source_log)
            return path, result.source.files, result.source.fingerprint

        artifact, _artifact_fingerprint = self.attempts.run(
            run_id,
            druid,
            "cocina",
            druid,
            SIGNATURES["cocina"],
            machine.fetch,
        )
        result = machine.result
        self._record_refresh(run_id, druid, cache, result, source_log)
        self.store.set_stage_output_fingerprint(
            druid, "cocina", result.source.fingerprint
        )
        return artifact, result.source.files, result.source.fingerprint

    def _record_fresh_cache(
        self,
        run_id: int,
        druid: str,
        cache: CocinaCache,
        result: CocinaRefreshResult,
        source_log: EventLog,
    ) -> None:
        source = result.source
        if (
            source.fingerprint != cache.stored_fingerprint
            or cache.inventory_signature != SOURCE_INVENTORY_SIGNATURE
        ):
            self._set_source(druid, result)
            source_log.write(
                run_id=run_id,
                druid=druid,
                stage="cocina",
                event="source_inventory_changed",
                source_fingerprint=source.fingerprint,
            )
        self.store.adopt_stage(
            druid,
            "cocina",
            druid,
            source.fingerprint,
            SIGNATURES["cocina"],
            cache.path,
        )
        source_log.write(
            run_id=run_id,
            druid=druid,
            stage="cocina",
            event="source_refresh_skipped",
            source_checked_at=cache.checked_at,
            max_age_days=COCINA_MAX_AGE.days,
        )

    def _record_refresh(
        self,
        run_id: int,
        druid: str,
        cache: CocinaCache,
        result: CocinaRefreshResult,
        source_log: EventLog,
    ) -> None:
        if result.state not in (
            CocinaRefreshState.NOT_MODIFIED,
            CocinaRefreshState.FETCHED,
        ):
            raise RuntimeError(f"Unexpected refresh state: {result.state.value}")
        source_changed = self._set_source(druid, result)
        if result.state == CocinaRefreshState.NOT_MODIFIED:
            source_log.write(
                run_id=run_id,
                druid=druid,
                stage="cocina",
                event="source_not_modified",
                etag=result.validators.etag or cache.validators.etag,
                source_changed=source_changed,
            )
            return
        source_log.write(
            run_id=run_id,
            druid=druid,
            stage="cocina",
            event="source_fetched",
            etag=result.validators.etag,
            source_fingerprint=result.source.fingerprint,
            source_changed=source_changed,
        )

    def _set_source(self, druid: str, result: CocinaRefreshResult) -> bool:
        source = result.source
        return self.store.set_source(
            druid,
            source.fingerprint,
            str(source.data.get("version")),
            source.files,
            etag=result.validators.etag,
            last_modified=result.validators.last_modified,
            cache_sha256=source.cache_sha256,
        )

    def create_search_metadata(self, druid: str, version_dir: Path) -> Path:
        input_dir = version_dir / "traject-input"
        input_dir.mkdir(exist_ok=True)
        shutil.copy2(version_dir / "cocina.json", input_dir / f"{druid}.json")
        env = os.environ.copy()
        env["PURL_DATA_DIR"] = str(input_dir)
        try:
            result = subprocess.run(
                [
                    "bundle",
                    "exec",
                    "traject",
                    "-c",
                    str(self.settings.root / "sdr_config.rb"),
                ],
                cwd=self.settings.root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise StageError(
                "bundle executable is not installed; install Ruby and run `bundle install`"
            ) from exc
        if result.returncode:
            detail = result.stderr[-2000:]
            if any(
                token in result.stderr.lower()
                for token in (
                    "connectionfailed",
                    "timed out",
                    "timeout",
                    "temporary failure",
                    "getaddrinfo",
                )
            ):
                raise TransientStageError(f"Traject dependency failed: {detail}")
            raise StageError(f"Traject failed: {detail}")
        try:
            metadata = json.loads(result.stdout)
        except ValueError as exc:
            raise StageError(
                f"Traject output was not one valid JSON record: {exc}"
            ) from exc
        output = version_dir / "metadata.json"
        output.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        shutil.rmtree(input_dir)
        return output
