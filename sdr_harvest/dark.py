from __future__ import annotations

import csv
import fcntl
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .core import StageError
from .manifests import DRUID_RE, read_druids
from .state import StateStore


DARK_LIST_FIELDS = ["druid", "darkened_on"]


def normalize_druid(value: str) -> str:
    match = DRUID_RE.match(value.strip())
    if not match:
        raise StageError(f"Invalid DRUID: {value!r}")
    return match.group(1).lower()


def read_dark_list(path: Path | None) -> set[str]:
    """Return disavowed DRUIDs; a missing list means no object is dark."""
    if path is None or not path.exists():
        return set()
    return read_druids(path)


def record_dark(path: Path, druids: list[str], darkened_on: str) -> list[str]:
    """Add DRUIDs to the dark list, keeping the original date of known ones."""
    rows: dict[str, str] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                rows[normalize_druid(row["druid"])] = row.get("darkened_on") or ""
    added = [druid for druid in sorted(set(druids)) if druid not in rows]
    if not added:
        return []
    rows.update((druid, darkened_on) for druid in added)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(DARK_LIST_FIELDS)
        writer.writerows(sorted(rows.items()))
    temporary.replace(path)
    return added


def remove_from_manifest(path: Path, druids: set[str]) -> int:
    """Drop manifest rows naming a dark DRUID, preserving every other row."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))

    def is_dark(row: list[str]) -> bool:
        for value in row:
            match = DRUID_RE.match(value.strip())
            if match:
                return match.group(1).lower() in druids
        return False

    kept = [row for row in rows if not is_dark(row)]
    removed = len(rows) - len(kept)
    if removed:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(kept)
        temporary.replace(path)
    return removed


def purge_local_state(store: StateStore, state_dir: Path, druid: str) -> bool:
    """Delete every database row, cached source, artifact, and log for an object."""
    paths = [state_dir / "sources" / druid, state_dir / "versions" / druid]
    logs = state_dir / "logs"
    if logs.exists():
        paths.extend(logs.glob(f"*/{druid}"))
        paths.extend(logs.glob(f"publish/*/{druid}.jsonl"))
    found = store.forget(druid)
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
            found = True
        elif path.exists():
            path.unlink()
            found = True
    return found


@contextmanager
def exclusive_state(state_dir: Path) -> Iterator[None]:
    """Hold the run lock so no build or publish recreates purged state."""
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = (state_dir / "run.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise StageError("Another sdr-harvest run or publish is already active") from exc
    try:
        yield
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()
