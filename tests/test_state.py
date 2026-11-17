import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sdr_harvest.attempts import StageAttempts
from sdr_harvest.core import (
    ALTO_EXTRACT_SIGNATURE,
    SIGNATURES,
    SOURCE_INVENTORY_SIGNATURE,
    Settings,
    TransientStageError,
    file_sha256,
)
from sdr_harvest.manifests import (
    cocina_pdf_files,
    cocina_source_files,
    source_fingerprint,
)
from sdr_harvest.pipeline import Pipeline
from sdr_harvest.services import PipelineServices
from sdr_harvest.state import STAGES, StateStore
from tests.support import DRUID, cocina


class StateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.store = StateStore(self.path / "state.sqlite3")
        self.store.reconcile_manifest({DRUID})

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_source_change_invalidates_downstream_success(self):
        self.store.set_source(DRUID, "old", "1", [])
        for stage in STAGES[1:]:
            artifact = self.path / stage
            artifact.write_text(stage)
            self.store.adopt_stage(DRUID, stage, "in", "out", "sig", artifact)
        changed = self.store.set_source(DRUID, "new", "2", [])
        self.assertTrue(changed)
        self.assertTrue(
            all(
                self.store.stage(DRUID, stage).status == "pending"
                for stage in STAGES[1:]
            )
        )

    def test_same_source_does_not_invalidate(self):
        self.store.set_source(DRUID, "same", "1", [])
        artifact = self.path / "download"
        artifact.write_text("ok")
        self.store.adopt_stage(DRUID, "download", "in", "out", "sig", artifact)
        self.assertFalse(self.store.set_source(DRUID, "same", "1", []))
        self.assertEqual("succeeded", self.store.stage(DRUID, "download").status)

    def test_managed_paths_are_stored_relative_to_state_directory(self):
        artifact = self.path / "versions" / DRUID / "source" / "solr.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("{}")
        self.store.adopt_stage(DRUID, "document", "in", "out", "sig", artifact)
        self.store.mark_built(DRUID, str(artifact.parent))

        stored = self.store.db.execute(
            "SELECT artifact_path FROM stage_state WHERE druid=? AND stage='document'",
            (DRUID,),
        ).fetchone()[0]
        current = self.store.object_row(DRUID)["current_artifact_dir"]

        self.assertEqual(str(Path("versions") / DRUID / "source" / "solr.json"), stored)
        self.assertEqual(str(Path("versions") / DRUID / "source"), current)
        self.assertEqual(artifact.resolve(), self.store.resolve_path(stored))

    def test_legacy_absolute_paths_are_migrated_after_state_directory_moves(self):
        self.store.close()
        old_root = Path("/old-machine/project/.sdr-harvest")
        connection = sqlite3.connect(self.path / "state.sqlite3")
        connection.execute(
            "DELETE FROM app_metadata WHERE key='relative-state-paths-v1'"
        )
        connection.execute(
            """INSERT INTO stage_state(
                 druid,stage,status,artifact_path,attempt_count
               ) VALUES(?,?,'succeeded',?,0)""",
            (
                DRUID,
                "document",
                str(old_root / "versions" / DRUID / "source" / "solr.json"),
            ),
        )
        connection.execute(
            "UPDATE objects SET current_artifact_dir=? WHERE druid=?",
            (str(old_root / "versions" / DRUID / "source"), DRUID),
        )
        connection.commit()
        connection.close()

        self.store = StateStore(self.path / "state.sqlite3")

        record = self.store.stage(DRUID, "document")
        self.assertEqual(
            str(Path("versions") / DRUID / "source" / "solr.json"),
            record.artifact_path,
        )
        self.assertEqual(
            (self.path / "versions" / DRUID / "source" / "solr.json").resolve(),
            self.store.resolve_path(record.artifact_path),
        )

    def test_relative_artifact_path_survives_state_directory_move(self):
        self.store.close()
        original = self.path / "original-state"
        moved = self.path / "moved-state"
        store = StateStore(original / "state.sqlite3")
        store.reconcile_manifest({DRUID})
        artifact = original / "versions" / DRUID / "source" / "solr.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("{}")
        store.adopt_stage(DRUID, "document", "in", "out", "sig", artifact)
        store.close()

        original.rename(moved)
        self.store = StateStore(moved / "state.sqlite3")
        record = self.store.stage(DRUID, "document")

        self.assertEqual(
            (moved / "versions" / DRUID / "source" / "solr.json").resolve(),
            self.store.resolve_path(record.artifact_path),
        )
        self.assertTrue(self.store.resolve_path(record.artifact_path).exists())

    def test_invalidate_marks_selected_stage_and_following_stages(self):
        for stage in STAGES:
            artifact = self.path / stage
            artifact.write_text(stage)
            self.store.adopt_stage(DRUID, stage, "in", "out", "sig", artifact)
        self.store.invalidate(DRUID, "chunk")
        self.assertEqual("succeeded", self.store.stage(DRUID, "extract").status)
        self.assertEqual("pending", self.store.stage(DRUID, "chunk").status)
        self.assertEqual("pending", self.store.stage(DRUID, "document").status)

    def test_existing_database_is_migrated_with_current_fingerprint_columns(self):
        legacy_path = self.path / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
        connection.execute(
            """CREATE TABLE objects (
               druid TEXT PRIMARY KEY, manifest_present INTEGER NOT NULL DEFAULT 1,
               source_fingerprint TEXT, source_version TEXT, current_artifact_dir TEXT,
               published_fingerprint TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE publications (
               druid TEXT NOT NULL REFERENCES objects(druid) ON DELETE CASCADE,
               target_url TEXT NOT NULL, source_fingerprint TEXT NOT NULL,
               status TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
               started_at TEXT, finished_at TEXT, error_category TEXT,
               error_message TEXT, receipt_path TEXT,
               PRIMARY KEY (druid, target_url)
            )"""
        )
        connection.commit()
        connection.close()
        migrated = StateStore(legacy_path)
        columns = {row[1] for row in migrated.db.execute("PRAGMA table_info(objects)")}
        self.assertTrue(
            {
                "source_etag",
                "source_last_modified",
                "source_checked_at",
                "source_cache_sha256",
                "source_inventory_signature",
            }.issubset(columns)
        )
        publication_columns = {
            row[1] for row in migrated.db.execute("PRAGMA table_info(publications)")
        }
        self.assertIn("document_fingerprint", publication_columns)
        migrated.close()


class RetryTest(unittest.TestCase):
    def test_transient_failure_is_retried_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            run_id = store.start_run("manifest.csv")
            attempts = StageAttempts(Settings(root, root, max_retries=2), store)
            artifact = root / "result"
            operation = Mock(side_effect=[TransientStageError("later"), artifact])
            artifact.write_text("ok")
            with patch("sdr_harvest.attempts.time.sleep"):
                result, _ = attempts.run(
                    run_id, DRUID, "download", "input", "signature", operation
                )
            self.assertEqual(artifact, result)
            self.assertEqual(2, operation.call_count)
            self.assertEqual("succeeded", store.stage(DRUID, "download").status)
            attempts = store.db.execute(
                "SELECT status,transient FROM attempts ORDER BY id"
            ).fetchall()
            self.assertEqual(
                [("failed", 1), ("succeeded", None)], [tuple(row) for row in attempts]
            )
            store.close()

    def test_retry_after_overrides_exponential_delay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            run_id = store.start_run("manifest.csv")
            attempts = StageAttempts(Settings(root, root, max_retries=2), store)
            artifact = root / "result"
            artifact.write_text("ok")
            operation = Mock(
                side_effect=[
                    TransientStageError("rate limited", retry_after=30),
                    artifact,
                ]
            )

            with (
                patch("sdr_harvest.attempts.random.random", return_value=0),
                patch("sdr_harvest.attempts.time.sleep") as sleep,
            ):
                attempts.run(run_id, DRUID, "download", "input", "signature", operation)

            sleep.assert_called_once_with(30)
            events = [
                json.loads(line)
                for line in (root / "logs" / str(run_id) / DRUID / "download.jsonl")
                .read_text()
                .splitlines()
            ]
            retrying = next(event for event in events if event["event"] == "retrying")
            self.assertEqual(30, retrying["retry_after_seconds"])
            self.assertEqual(30, retrying["delay_seconds"])
            store.close()


class ResumeTest(unittest.TestCase):
    def _seed_current_extraction(
        self, root: Path, store: StateStore, *, use_alto: bool
    ) -> None:
        source_fp = "source-fingerprint"
        cache = root / "sources" / DRUID / "cocina.json"
        cache.parent.mkdir(parents=True)
        cocina_data = {
            "type": (
                "https://cocina.sul.stanford.edu/models/book"
                if use_alto
                else "https://cocina.sul.stanford.edu/models/document"
            )
        }
        cache.write_text(json.dumps(cocina_data))
        store.set_source(
            DRUID,
            source_fp,
            "1",
            [],
            cache_sha256=file_sha256(cache),
        )
        version_dir = root / "versions" / DRUID / source_fp
        source_dir = version_dir / "pdfs"
        source_dir.mkdir(parents=True)
        version_dir.joinpath("cocina.json").write_text(json.dumps(cocina_data))
        source_dir.joinpath("page.xml" if use_alto else "document.pdf").write_text(
            "<alto/>" if use_alto else "pdf"
        )
        input_fp = source_fp
        for stage in STAGES[1:]:
            artifact = version_dir / f"{stage}.artifact"
            artifact.write_text(stage)
            output_fp = f"output-{stage}"
            signature = (
                ALTO_EXTRACT_SIGNATURE if stage == "extract" else SIGNATURES[stage]
            )
            store.adopt_stage(DRUID, stage, input_fp, output_fp, signature, artifact)
            input_fp = output_fp
        store.mark_built(DRUID, str(version_dir))

    def test_pymupdf_signature_change_does_not_queue_alto_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            self._seed_current_extraction(root, store, use_alto=True)

            estimate, pending = Pipeline(Settings(root, root), store)._plan_work(
                [DRUID], show_progress=False
            )

            self.assertEqual([], pending)
            self.assertTrue(all(count == 0 for count in estimate.values()))
            store.close()

    def test_pymupdf_signature_change_queues_pdf_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            self._seed_current_extraction(root, store, use_alto=False)
            store.db.execute(
                "UPDATE stage_state SET stage_signature=? "
                "WHERE druid=? AND stage='extract'",
                (ALTO_EXTRACT_SIGNATURE, DRUID),
            )
            store.db.commit()

            estimate, pending = Pipeline(Settings(root, root), store)._plan_work(
                [DRUID], show_progress=False
            )

            self.assertEqual([DRUID], pending)
            self.assertEqual(1, estimate["extract"])
            store.close()

    def test_changed_source_inventory_policy_queues_cached_object_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            source = cocina()
            files = cocina_source_files(source)
            source_fp = source_fingerprint(source, files)
            cache = root / "sources" / DRUID / "cocina.json"
            cache.parent.mkdir(parents=True)
            cache.write_text(json.dumps(source, sort_keys=True))
            store.set_source(
                DRUID,
                source_fp,
                "1",
                files,
                cache_sha256=file_sha256(cache),
            )
            self.assertEqual(
                SOURCE_INVENTORY_SIGNATURE,
                store.object_row(DRUID)["source_inventory_signature"],
            )
            store.db.execute(
                "UPDATE objects SET source_inventory_signature=NULL WHERE druid=?",
                (DRUID,),
            )
            store.db.commit()
            input_fp = source_fp
            version_dir = root / "versions" / DRUID / source_fp
            version_dir.mkdir(parents=True)
            for stage in STAGES[1:]:
                artifact = version_dir / stage
                artifact.write_text(stage)
                output_fp = f"output-{stage}"
                store.adopt_stage(
                    DRUID,
                    stage,
                    input_fp,
                    output_fp,
                    SIGNATURES[stage],
                    artifact,
                )
                input_fp = output_fp

            estimate, pending = Pipeline(Settings(root, root), store)._plan_work(
                [DRUID], show_progress=False
            )

            self.assertEqual([DRUID], pending)
            self.assertEqual(1, estimate["cocina"])
            self.assertTrue(
                all(estimate[stage] == 0 for stage in STAGES if stage != "cocina")
            )
            store.close()

    def test_fully_current_objects_are_not_queued_for_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            source = cocina()
            files = cocina_pdf_files(source)
            source_fp = source_fingerprint(source, files)
            cache = root / "sources" / DRUID / "cocina.json"
            cache.parent.mkdir(parents=True)
            cache.write_text(json.dumps(source, sort_keys=True))
            store.set_source(
                DRUID,
                source_fp,
                "1",
                files,
                cache_sha256=file_sha256(cache),
            )
            input_fp = source_fp
            version_dir = root / "versions" / DRUID / source_fp
            version_dir.mkdir(parents=True)
            for stage in STAGES[1:]:
                artifact = version_dir / stage
                artifact.write_text(stage)
                output_fp = f"output-{stage}"
                store.adopt_stage(
                    DRUID,
                    stage,
                    input_fp,
                    output_fp,
                    SIGNATURES[stage],
                    artifact,
                )
                input_fp = output_fp

            estimate, pending = Pipeline(Settings(root, root), store)._plan_work(
                [DRUID], show_progress=False
            )

            self.assertEqual([], pending)
            self.assertTrue(all(count == 0 for count in estimate.values()))
            store.close()

    def test_current_artifacts_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            source = cocina()
            files = cocina_pdf_files(source)
            source_fp = source_fingerprint(source, files)
            store.set_source(DRUID, source_fp, "1", files)
            version_dir = root / "versions" / DRUID / source_fp
            version_dir.mkdir(parents=True)
            cocina_path = root / "source.json"
            cocina_path.write_text(json.dumps(source))
            input_fp = source_fp
            for stage in STAGES[1:]:
                artifact = version_dir / stage
                artifact.write_text(stage)
                output_fp = f"output-{stage}"
                store.adopt_stage(
                    DRUID, stage, input_fp, output_fp, SIGNATURES[stage], artifact
                )
                input_fp = output_fp
            progress_events = []
            closer = Mock()
            services = PipelineServices(
                attempts=Mock(),
                metadata=Mock(),
                downloader=Mock(),
                extractor=Mock(),
                chunker=Mock(),
                embedder=Mock(),
                document_builder=Mock(),
                closer=closer,
            )
            service_factory = Mock()
            service_factory.create.return_value = services
            pipeline = Pipeline(
                Settings(root, root),
                store,
                progress_callback=lambda druid, stage, event: progress_events.append(
                    (druid, stage, event)
                ),
                service_factory=service_factory,
            )
            services.metadata.fetch_cocina.return_value = (
                cocina_path,
                files,
                source_fp,
            )
            services.extractor.signature.return_value = SIGNATURES["extract"]

            pipeline.run_object(store.start_run("manifest.csv"), DRUID)

            service_factory.create.assert_called_once_with(
                pipeline.settings, store, None
            )
            closer.assert_called_once_with()
            self.assertEqual(
                str(Path("versions") / DRUID / source_fp),
                store.object_row(DRUID)["current_artifact_dir"],
            )
            services.attempts.run.assert_not_called()
            self.assertIn((DRUID, "cocina", "started"), progress_events)
            self.assertIn((DRUID, "cocina", "succeeded"), progress_events)
            self.assertIn((DRUID, "document", "skipped"), progress_events)
            estimate = pipeline._estimate_work([DRUID], show_progress=False)
            self.assertEqual(1, estimate["cocina"])
            self.assertTrue(all(estimate[stage] == 0 for stage in STAGES[1:]))

            store.invalidate(DRUID, "chunk")
            estimate = pipeline._estimate_work([DRUID], show_progress=False)
            self.assertEqual(0, estimate["extract"])
            self.assertEqual(1, estimate["chunk"])
            self.assertEqual(1, estimate["embed"])
            self.assertEqual(1, estimate["document"])
            store.close()
