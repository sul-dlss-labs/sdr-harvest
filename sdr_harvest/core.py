from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from .stages import ALTO_EXTRACT_SIGNATURE, PDF_EXTRACT_SIGNATURE, SIGNATURES


SOURCE_INVENTORY_SIGNATURE = "prefer-complete-book-alto-v2-indexable-access"


class StageError(RuntimeError):
    transient = False


class TransientStageError(StageError):
    transient = True

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_fingerprint(path: Path) -> str:
    """Identify a directory artifact by its relative file names and contents."""
    return fingerprint(
        sorted(
            (str(item.relative_to(path)), file_sha256(item))
            for item in path.rglob("*")
            if item.is_file()
        )
    )


def previous_versions(version_dir: Path) -> list[Path]:
    """Return sibling artifact directories for the same object, newest first."""
    if not version_dir.parent.exists():
        return []
    siblings = [
        path
        for path in version_dir.parent.iterdir()
        if path.is_dir() and path != version_dir
    ]
    return sorted(siblings, key=lambda path: path.stat().st_mtime, reverse=True)


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_exception(exc: Exception) -> tuple[bool, str]:
    if isinstance(exc, TransientStageError):
        return True, type(exc).__name__
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True, type(exc).__name__
    return False, type(exc).__name__


@dataclass
class Settings:
    root: Path
    state_dir: Path
    solr_url: str = "http://localhost:8983/solr/sdr-search"
    verify_tls: bool = True
    workers: int = 4
    max_retries: int = 5
    keep_failed_downloads: bool = False
    publish_batch_size: int = 25
    publish_max_batch_bytes: int = 25 * 1024 * 1024
    dark_list: Path | None = None

    @classmethod
    def from_root(cls, root: Path, **kwargs) -> "Settings":
        return cls(root=root, state_dir=root / ".sdr-harvest", **kwargs)


class EventLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, **event: object) -> None:
        event["timestamp"] = datetime.now(UTC).isoformat()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, default=str) + "\n")


@contextmanager
def interruptible_thread_pool(max_workers: int):
    """Do not wait for active worker threads after a keyboard interrupt."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield executor
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def exit_with_parent(parent_pid: int, interval: float = 1.0) -> None:
    """Process pool initializer: end this worker once its coordinator is gone.

    A worker otherwise outlives a coordinator killed without cleanup (for
    example by ``kill -9``), re-parented to init and holding its memory.
    """

    def watch() -> None:
        while os.getppid() == parent_pid:
            time.sleep(interval)
        os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def terminate_workers(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    """Cancel queued work and stop workers, including any mid-extraction."""
    # Captured before shutdown, which may discard the executor's process table.
    processes = list((executor._processes or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


def _init_pool_worker(parent_pid: int, initializer, initargs: tuple) -> None:
    exit_with_parent(parent_pid)
    if initializer is not None:
        initializer(*initargs)


@contextmanager
def interruptible_process_pool(
    max_workers: int,
    *,
    mp_context=None,
    initializer=None,
    initargs: tuple = (),
):
    """Run CPU-bound work in isolated processes without blocking Ctrl-C exit.

    If the run is interrupted or fails, running workers are terminated rather
    than left to finish their current task after the coordinator exits.
    """
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context or multiprocessing.get_context("spawn"),
        initializer=_init_pool_worker,
        initargs=(os.getpid(), initializer, initargs),
    )
    try:
        yield executor
    except BaseException:
        # Workers ignore SIGINT and would otherwise finish their current task
        # after the coordinator exits, so stop them now.
        terminate_workers(executor)
        raise
    else:
        executor.shutdown(wait=True)
