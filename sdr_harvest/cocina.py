from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Protocol

import requests

from .core import StageError, TransientStageError, file_sha256
from .manifests import cocina_source_files, source_fingerprint


COCINA_MAX_AGE = timedelta(days=7)


class StoredCocina(Protocol):
    def __getitem__(self, key: str) -> object: ...


def _stored_value(stored: StoredCocina | None, key: str) -> str | None:
    if stored is None:
        return None
    value = stored[key]
    return str(value) if value else None


class CocinaRefreshState(Enum):
    CHECK_CACHE = "check_cache"
    REFRESH_REQUIRED = "refresh_required"
    FRESH_CACHE = "fresh_cache"
    NOT_MODIFIED = "not_modified"
    FETCHED = "fetched"


@dataclass(frozen=True)
class CocinaSource:
    data: dict
    files: list[dict]
    fingerprint: str
    cache_sha256: str


@dataclass(frozen=True)
class CocinaValidators:
    etag: str | None
    last_modified: str | None


@dataclass(frozen=True)
class CocinaRefreshResult:
    state: CocinaRefreshState
    source: CocinaSource
    validators: CocinaValidators


@dataclass(frozen=True)
class CocinaCache:
    path: Path
    valid: bool
    stored_fingerprint: str | None
    checked_at: str | None
    inventory_signature: str | None
    validators: CocinaValidators

    @classmethod
    def inspect(
        cls, path: Path, stored: StoredCocina | None
    ) -> CocinaCache:
        stored_fingerprint = _stored_value(stored, "source_fingerprint")
        stored_sha256 = _stored_value(stored, "source_cache_sha256")
        valid = bool(
            stored_fingerprint
            and stored_sha256
            and path.exists()
            and file_sha256(path) == stored_sha256
        )
        return cls(
            path=path,
            valid=valid,
            stored_fingerprint=stored_fingerprint,
            checked_at=_stored_value(stored, "source_checked_at"),
            inventory_signature=_stored_value(
                stored, "source_inventory_signature"
            ),
            validators=CocinaValidators(
                etag=_stored_value(stored, "source_etag"),
                last_modified=_stored_value(stored, "source_last_modified"),
            ),
        )

    def read(self) -> CocinaSource:
        data = json.loads(self.path.read_text())
        files = cocina_source_files(data)
        return CocinaSource(
            data=data,
            files=files,
            fingerprint=source_fingerprint(data, files),
            cache_sha256=file_sha256(self.path),
        )

    def request_headers(self) -> dict[str, str]:
        if not self.valid:
            return {}
        if self.validators.etag:
            return {"If-None-Match": self.validators.etag}
        if self.validators.last_modified:
            return {"If-Modified-Since": self.validators.last_modified}
        return {}


def cocina_checked_recently(checked_at: str | None) -> bool:
    """Return whether a real PURL check occurred within the freshness window."""
    if not checked_at:
        return False
    try:
        checked = datetime.fromisoformat(checked_at)
    except ValueError:
        return False
    if checked.tzinfo is None:
        checked = checked.replace(tzinfo=UTC)
    return checked >= datetime.now(UTC) - COCINA_MAX_AGE


class CocinaRefreshMachine:
    """Transition a COCINA cache to a current terminal state."""

    def __init__(
        self,
        druid: str,
        cache: CocinaCache,
        http: requests.Session,
    ) -> None:
        self.druid = druid
        self.cache = cache
        self.http = http
        self.state = CocinaRefreshState.CHECK_CACHE
        self._result: CocinaRefreshResult | None = None

    @property
    def result(self) -> CocinaRefreshResult:
        if self._result is None:
            raise RuntimeError("COCINA refresh has not reached a terminal state")
        return self._result

    def use_fresh_cache(self) -> CocinaRefreshResult | None:
        if self.state != CocinaRefreshState.CHECK_CACHE:
            raise RuntimeError(f"Cannot check cache from state {self.state.value}")
        if self.cache.valid and cocina_checked_recently(self.cache.checked_at):
            return self._finish(
                CocinaRefreshState.FRESH_CACHE,
                self.cache.read(),
                self.cache.validators,
            )
        self.state = CocinaRefreshState.REFRESH_REQUIRED
        return None

    def fetch(self) -> Path:
        """Fetch one refresh attempt; transient failures remain retryable."""
        if self.state != CocinaRefreshState.REFRESH_REQUIRED:
            raise RuntimeError(f"Cannot refresh COCINA from state {self.state.value}")
        response = self.http.get(
            f"https://purl.stanford.edu/{self.druid}.json",
            headers=self.cache.request_headers(),
            timeout=60,
        )
        if response.status_code == 304:
            return self._accept_not_modified(response)
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientStageError(f"PURL HTTP {response.status_code}")
        if response.status_code != 200:
            raise StageError(f"PURL HTTP {response.status_code}")
        return self._accept_fetched(response)

    def _accept_not_modified(self, response: requests.Response) -> Path:
        if not self.cache.valid:
            raise StageError("PURL returned 304 but no valid cached COCINA exists")
        self._finish(
            CocinaRefreshState.NOT_MODIFIED,
            self.cache.read(),
            CocinaValidators(
                response.headers.get("ETag"),
                response.headers.get("Last-Modified"),
            ),
        )
        return self.cache.path

    def _accept_fetched(self, response: requests.Response) -> Path:
        try:
            data = response.json()
        except ValueError as exc:
            raise StageError(f"Invalid COCINA JSON: {exc}") from exc
        if data.get("externalIdentifier") != f"druid:{self.druid}":
            raise StageError(
                "COCINA externalIdentifier does not match requested DRUID"
            )
        temporary = self.cache.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        temporary.replace(self.cache.path)
        files = cocina_source_files(data)
        self._finish(
            CocinaRefreshState.FETCHED,
            CocinaSource(
                data=data,
                files=files,
                fingerprint=source_fingerprint(data, files),
                cache_sha256=file_sha256(self.cache.path),
            ),
            CocinaValidators(
                response.headers.get("ETag"),
                response.headers.get("Last-Modified"),
            ),
        )
        return self.cache.path

    def _finish(
        self,
        state: CocinaRefreshState,
        source: CocinaSource,
        validators: CocinaValidators,
    ) -> CocinaRefreshResult:
        self.state = state
        self._result = CocinaRefreshResult(state, source, validators)
        return self._result
