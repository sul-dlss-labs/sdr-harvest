from __future__ import annotations

import concurrent.futures
import fcntl
import multiprocessing
import os
import queue
import shutil
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm

from .attempts import StageAttempts
from .chunk import Chunker
from .core import (
    EventLog,
    Settings,
    StageError,
    interruptible_process_pool,
)
from .create_solr_document import SolrDocumentBuilder
from .download import FileDownloader
from .embed import Embedder, create_shared_limits, install_shared_limits
from .extract_text import TextExtractor
from .dark import read_dark_list
from .manifests import parse_manifest
from .metadata import MetadataFetcher
from .planner import PipelinePlanner, artifact_belongs_to_version
from .stages import BUILD_STAGES
from .state import StateStore


_worker_events = None


def _init_object_worker(events, embedding_slots, rate_limit_open_at) -> None:
    """Prepare a worker process to build objects for the coordinator."""
    # The coordinator handles Ctrl-C and terminates workers itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _worker_events
    _worker_events = events
    install_shared_limits(embedding_slots, rate_limit_open_at)


def _build_object(
    settings: Settings, store_path: Path, run_id: int, druid: str
) -> tuple[str, str | None]:
    """Build one object in a worker process; errors return as text to pickle safely."""
    store = StateStore(store_path)
    try:
        Pipeline(
            settings,
            store,
            progress_callback=lambda item, stage, event: _worker_events.put(
                (item, stage, event)
            ),
        ).run_object(run_id, druid)
        return druid, None
    except Exception as exc:
        return druid, str(exc) or type(exc).__name__
    finally:
        store.close()


class Pipeline:
    """Coordinate resumable object builds without contacting Solr."""

    def __init__(
        self,
        settings: Settings,
        store: StateStore,
        progress_callback: Callable[[str, str, str], None] | None = None,
        pdf_executor: concurrent.futures.Executor | None = None,
    ):
        self.settings = settings
        self.store = store
        self.http = requests.Session()
        self.progress_callback = progress_callback
        self.pdf_executor = pdf_executor

    def _progress(self, druid: str, stage: str, event: str) -> None:
        if self.progress_callback:
            self.progress_callback(druid, stage, event)

    def run(
        self,
        manifest: Path,
        *,
        only: set[str] | None = None,
        show_progress: bool = True,
    ) -> dict:
        """Build every selected manifest object without contacting Solr."""
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        lock_stream = (self.settings.state_dir / "run.lock").open("w")
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_stream.close()
            raise StageError("Another sdr-harvest run is already active") from exc
        try:
            return self._run_locked(
                manifest, only=only, show_progress=show_progress
            )
        finally:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)
            lock_stream.close()

    def _estimate_work(
        self, druids: list[str], *, show_progress: bool = True
    ) -> dict[str, int]:
        """Conservatively estimate stage executions if sources are unchanged."""
        counts, _ = self._plan_work(druids, show_progress=show_progress)
        return counts

    def _plan_work(
        self, druids: list[str], *, show_progress: bool = True
    ) -> tuple[dict[str, int], list[str]]:
        """Estimate stage work and select objects that need to enter the pool."""
        plan = PipelinePlanner(self.settings, self.store).plan(
            druids, show_progress=show_progress
        )
        return plan.counts, plan.pending_druids

    def _run_locked(
        self,
        manifest: Path,
        *,
        only: set[str] | None = None,
        show_progress: bool = True,
    ) -> dict:
        druids = parse_manifest(manifest) - read_dark_list(self.settings.dark_list)
        new, absent = self.store.reconcile_manifest(druids)
        selected = sorted(druids if only is None else druids & only)
        run_id = self.store.start_run(str(manifest.resolve()))
        summary = {
            "total": len(selected),
            "succeeded": 0,
            "failed": 0,
            "new": len(new),
            "absent": len(absent),
        }
        try:
            estimate, pending_druids = self._plan_work(
                selected, show_progress=show_progress
            )
            already_current = len(selected) - len(pending_druids)
            summary["already_current"] = already_current
            summary["succeeded"] = already_current
            print(
                "Estimated work if remote COCINA is unchanged: "
                + ", ".join(
                    f"{stage}={count:,}" for stage, count in estimate.items()
                ),
                flush=True,
            )
            print(
                "Changed COCINA records may increase downstream work during the run.",
                flush=True,
            )
            # Each object builds in its own process so CPU-bound stages
            # (chunking, embedding reuse, Solr JSON) are not serialized by the GIL.
            context = multiprocessing.get_context("spawn")
            progress_events = context.Queue()
            workers = max(1, min(self.settings.workers, len(pending_druids)))
            if pending_druids:
                print(f"Object build worker processes: {workers}", flush=True)
            with interruptible_process_pool(
                workers,
                mp_context=context,
                initializer=_init_object_worker,
                initargs=(progress_events, *create_shared_limits(context)),
            ) as executor:
                pending = {
                    executor.submit(
                        _build_object,
                        self.settings,
                        self.store.path,
                        run_id,
                        druid,
                    )
                    for druid in pending_druids
                }
                active: dict[str, str] = {}
                # Progress events travel separately from results, so a late
                # event must not mark a finished object active again.
                finished: set[str] = set()
                last_refresh = 0.0
                with tqdm(
                    total=len(selected),
                    initial=already_current,
                    desc="Processing pipeline objects",
                    unit="object",
                    disable=not show_progress,
                ) as progress:
                    while pending:
                        done, pending = concurrent.futures.wait(
                            pending,
                            timeout=0.25,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        while True:
                            try:
                                druid, stage, event = progress_events.get_nowait()
                            except queue.Empty:
                                break
                            if druid in finished:
                                continue
                            if event == "started":
                                active[druid] = stage
                            elif active.get(druid) == stage:
                                active.pop(druid, None)
                        for future in done:
                            try:
                                druid, error = future.result()
                            except concurrent.futures.process.BrokenProcessPool as exc:
                                raise StageError(
                                    "An object build worker process exited "
                                    "unexpectedly; rerun to resume"
                                ) from exc
                            finished.add(druid)
                            active.pop(druid, None)
                            if error is None:
                                summary["succeeded"] += 1
                            else:
                                summary["failed"] += 1
                                tqdm.write(f"FAIL {druid}: {error}")
                            progress.update(1)
                        active_counts = Counter(active.values())
                        activity = ",".join(
                            f"{stage}:{count}"
                            for stage, count in sorted(active_counts.items())
                        ) or "waiting"
                        progress.set_postfix_str(
                            f"remaining={len(pending):,} active={activity} "
                            f"ok={summary['succeeded']:,} failed={summary['failed']:,}",
                            refresh=False,
                        )
                        now = time.monotonic()
                        if done or now - last_refresh >= 1:
                            progress.refresh()
                            last_refresh = now
            self.store.finish_run(
                run_id, "failed" if summary["failed"] else "succeeded", summary
            )
        except BaseException:
            self.store.finish_run(run_id, "interrupted", summary)
            raise
        return summary

    def run_object(self, run_id: int, druid: str) -> None:
        """Run one object through every non-current build stage in order."""
        attempts = StageAttempts(self.settings, self.store)
        metadata = MetadataFetcher(
            self.settings, self.store, self.http, attempts
        )
        downloader = FileDownloader(
            self.http,
            keep_failed_downloads=self.settings.keep_failed_downloads,
            stacks_token=os.environ.get("STACKS_TOKEN"),
        )
        extractor = TextExtractor(pdf_executor=self.pdf_executor)
        chunker = Chunker()
        embedder = Embedder()
        document_builder = SolrDocumentBuilder()
        self._progress(druid, "cocina", "started")
        try:
            cocina_path, files, source_fp = metadata.fetch_cocina(run_id, druid)
        except Exception:
            self._progress(druid, "cocina", "failed")
            raise
        self._progress(druid, "cocina", "succeeded")
        version_dir = self.settings.state_dir / "versions" / druid / source_fp
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cocina_path, version_dir / "cocina.json")

        input_fp = source_fp
        outputs: dict[str, str] = {}
        # Artifacts recorded as current before this version, which the extract
        # and embed stages may reuse when their own inputs are unchanged.
        previous: dict[str, Path | None] = {}
        reuse_extract: Path | None = None
        operations: dict[str, Callable[[], Path]] = {
            "download": lambda: downloader.run(druid, files, version_dir),
            "metadata": lambda: metadata.create_search_metadata(druid, version_dir),
            "extract": lambda: extractor.run(
                version_dir, outputs["download"], reuse_from=reuse_extract
            ),
            "chunk": lambda: chunker.run(druid, version_dir),
            "embed": lambda: embedder.run(
                version_dir, legacy_output=previous.get("embed")
            ),
            "document": lambda: document_builder.run(druid, source_fp, version_dir),
        }
        for stage in BUILD_STAGES:
            operation = operations[stage.name]
            dynamic_signature = (
                extractor.signature(version_dir) if stage.dynamic_signature else None
            )
            signature = stage.signature(dynamic_signature)
            record = self.store.stage(druid, stage.name)
            artifact_is_current = bool(
                record
                and artifact_belongs_to_version(
                    self.store, record.artifact_path, version_dir
                )
            )
            is_current = stage.is_current(
                record,
                input_fp,
                signature,
                artifact_is_current=artifact_is_current,
            )
            if is_current:
                input_fp = record.output_fingerprint or input_fp
                outputs[stage.name] = input_fp
                self._progress(druid, stage.name, "skipped")
                continue
            previous[stage.name] = (
                self.store.resolve_path(record.artifact_path)
                if record
                and record.status == "succeeded"
                and record.stage_signature == signature
                and record.artifact_path
                else None
            )
            if stage.name == "extract":
                reuse_extract = extractor.reusable_output(
                    version_dir, outputs["download"], previous["extract"]
                )
                if reuse_extract is not None:
                    EventLog(
                        self.settings.state_dir
                        / "logs"
                        / str(run_id)
                        / druid
                        / "extract.jsonl"
                    ).write(
                        run_id=run_id,
                        druid=druid,
                        stage="extract",
                        event="reusing_previous_extraction",
                        reuse_from=str(reuse_extract),
                    )
            self._progress(druid, stage.name, "started")
            try:
                _, output_fp = attempts.run(
                    run_id, druid, stage.name, input_fp, signature, operation
                )
            except Exception:
                self._progress(druid, stage.name, "failed")
                raise
            self._progress(druid, stage.name, "succeeded")
            input_fp = output_fp
            outputs[stage.name] = input_fp
        self.store.mark_built(druid, str(version_dir))
