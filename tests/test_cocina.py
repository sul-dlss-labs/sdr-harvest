import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from sdr_harvest.attempts import StageAttempts
from sdr_harvest.cocina import (
    CocinaCache,
    CocinaRefreshMachine,
    CocinaRefreshState,
)
from sdr_harvest.core import (
    SIGNATURES,
    SOURCE_INVENTORY_SIGNATURE,
    Settings,
    TransientStageError,
    file_sha256,
)
from sdr_harvest.manifests import (
    cocina_pdf_files,
    source_fingerprint,
)
from sdr_harvest.metadata import MetadataFetcher
from sdr_harvest.pipeline import Pipeline
from sdr_harvest.publisher import (
    COMMIT_WITHIN_MS,
    BatchPublishError,
    CorpusPublisher,
    PublicationItem,
    SolrPublisher,
)
from sdr_harvest.state import StateStore
from tests.support import DRUID, cocina


class CocinaRefreshMachineTest(unittest.TestCase):
    def cache_record(self, path, *, fresh=False):
        return {
            "source_fingerprint": "stored",
            "source_cache_sha256": file_sha256(path),
            "source_checked_at": (
                datetime.now(UTC) - timedelta(days=1 if fresh else 8)
            ).isoformat(),
            "source_inventory_signature": SOURCE_INVENTORY_SIGNATURE,
            "source_etag": 'W/"old"',
            "source_last_modified": "Wed, 29 Jul 2026 19:04:31 GMT",
        }

    def test_fresh_cache_reaches_terminal_state_without_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cocina.json"
            path.write_text(json.dumps(cocina(), sort_keys=True))
            http = Mock()
            machine = CocinaRefreshMachine(
                DRUID,
                CocinaCache.inspect(path, self.cache_record(path, fresh=True)),
                http,
            )

            result = machine.use_fresh_cache()

            self.assertEqual(CocinaRefreshState.FRESH_CACHE, machine.state)
            self.assertIs(result, machine.result)
            self.assertEqual(cocina_pdf_files(cocina()), result.source.files)
            http.get.assert_not_called()

    def test_304_transitions_refresh_required_to_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cocina.json"
            path.write_text(json.dumps(cocina(), sort_keys=True))
            response = Mock(
                status_code=304,
                headers={"ETag": 'W/"old"'},
            )
            http = Mock()
            http.get.return_value = response
            machine = CocinaRefreshMachine(
                DRUID,
                CocinaCache.inspect(path, self.cache_record(path)),
                http,
            )

            self.assertIsNone(machine.use_fresh_cache())
            self.assertEqual(CocinaRefreshState.REFRESH_REQUIRED, machine.state)
            machine.fetch()

            self.assertEqual(CocinaRefreshState.NOT_MODIFIED, machine.state)
            self.assertEqual('W/"old"', machine.result.validators.etag)
            self.assertEqual(
                {"If-None-Match": 'W/"old"'},
                http.get.call_args.kwargs["headers"],
            )
            response.json.assert_not_called()

    def test_200_transitions_to_fetched_and_replaces_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cocina.json"
            changed = cocina(version=2, digest="changed")
            response = Mock(
                status_code=200,
                headers={"ETag": 'W/"new"'},
            )
            response.json.return_value = changed
            http = Mock()
            http.get.return_value = response
            machine = CocinaRefreshMachine(DRUID, CocinaCache.inspect(path, None), http)

            self.assertIsNone(machine.use_fresh_cache())
            machine.fetch()

            self.assertEqual(CocinaRefreshState.FETCHED, machine.state)
            self.assertEqual("changed", machine.result.source.files[0]["sha1"])
            self.assertEqual(changed, json.loads(path.read_text()))
            self.assertEqual({}, http.get.call_args.kwargs["headers"])

    def test_transient_failure_remains_refreshable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cocina.json"
            response = Mock(status_code=200, headers={})
            response.json.return_value = cocina()
            http = Mock()
            http.get.side_effect = [Mock(status_code=503), response]
            machine = CocinaRefreshMachine(DRUID, CocinaCache.inspect(path, None), http)
            machine.use_fresh_cache()

            with self.assertRaisesRegex(TransientStageError, "PURL HTTP 503"):
                machine.fetch()

            self.assertEqual(CocinaRefreshState.REFRESH_REQUIRED, machine.state)
            machine.fetch()
            self.assertEqual(CocinaRefreshState.FETCHED, machine.state)


class ConditionalCocinaTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = StateStore(self.root / "state.sqlite3")
        self.store.reconcile_manifest({DRUID})
        self.http = Mock()
        settings = Settings(self.root, self.root)
        self.metadata = MetadataFetcher(
            settings, self.store, self.http, StageAttempts(settings, self.store)
        )
        self.data = cocina()
        self.files = cocina_pdf_files(self.data)
        self.source_fp = source_fingerprint(self.data, self.files)
        self.cache = self.root / "sources" / DRUID / "cocina.json"
        self.cache.parent.mkdir(parents=True)
        self.cache.write_text(json.dumps(self.data, sort_keys=True))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def response(self, status, *, data=None, etag=None, last_modified=None):
        response = Mock(status_code=status)
        response.headers = {}
        if etag:
            response.headers["ETag"] = etag
        if last_modified:
            response.headers["Last-Modified"] = last_modified
        response.json = (
            Mock(side_effect=AssertionError("JSON must not be parsed"))
            if data is None
            else Mock(return_value=data)
        )
        return response

    def seed_validators(
        self,
        *,
        etag='W/"old"',
        last_modified="Wed, 29 Jul 2026 19:04:31 GMT",
        fresh=False,
    ):
        self.store.set_source(
            DRUID,
            self.source_fp,
            "1",
            self.files,
            etag=etag,
            last_modified=last_modified,
            cache_sha256=file_sha256(self.cache),
        )
        checked_at = datetime.now(UTC) - timedelta(days=1 if fresh else 8)
        self.store.db.execute(
            "UPDATE objects SET source_checked_at=? WHERE druid=?",
            (checked_at.isoformat(), DRUID),
        )
        self.store.db.commit()

    def test_recent_valid_cache_skips_purl_without_sliding_freshness_window(self):
        self.seed_validators(fresh=True)
        checked_at = self.store.object_row(DRUID)["source_checked_at"]
        self.http.get = Mock(side_effect=AssertionError("PURL must not be called"))

        path, files, source_fp = self.metadata.fetch_cocina(
            self.store.start_run("manifest.csv"), DRUID
        )

        self.assertEqual(self.cache, path)
        self.assertEqual(self.files, files)
        self.assertEqual(self.source_fp, source_fp)
        self.http.get.assert_not_called()
        self.assertEqual(checked_at, self.store.object_row(DRUID)["source_checked_at"])
        self.assertEqual(
            0,
            self.store.db.execute(
                "SELECT count(*) FROM attempts WHERE stage='cocina'"
            ).fetchone()[0],
        )
        estimate = Pipeline(Settings(self.root, self.root), self.store)._estimate_work(
            [DRUID], show_progress=False
        )
        self.assertEqual(0, estimate["cocina"])

    def test_etag_304_reuses_stored_fingerprint_and_files_without_parsing(self):
        self.seed_validators()
        response = self.response(
            304, etag='W/"old"', last_modified="Wed, 29 Jul 2026 19:04:31 GMT"
        )
        self.http.get = Mock(return_value=response)
        path, files, source_fp = self.metadata.fetch_cocina(
            self.store.start_run("manifest.csv"), DRUID
        )
        self.assertEqual(self.cache, path)
        self.assertEqual(self.files, files)
        self.assertEqual(self.source_fp, source_fp)
        self.assertEqual(
            'W/"old"', self.http.get.call_args.kwargs["headers"]["If-None-Match"]
        )
        response.json.assert_not_called()
        self.assertIsNotNone(self.store.object_row(DRUID)["source_checked_at"])

    def test_last_modified_is_used_when_etag_is_unavailable(self):
        self.seed_validators(etag=None)
        response = self.response(304, last_modified="Wed, 29 Jul 2026 19:04:31 GMT")
        self.http.get = Mock(return_value=response)
        self.metadata.fetch_cocina(self.store.start_run("manifest.csv"), DRUID)
        headers = self.http.get.call_args.kwargs["headers"]
        self.assertEqual(
            {"If-Modified-Since": "Wed, 29 Jul 2026 19:04:31 GMT"}, headers
        )
        response.json.assert_not_called()

    def test_changed_response_is_parsed_and_updates_validators(self):
        self.seed_validators()
        changed = cocina(version=2, digest="changed")
        response = self.response(
            200,
            data=changed,
            etag='W/"new"',
            last_modified="Thu, 30 Jul 2026 19:04:31 GMT",
        )
        self.http.get = Mock(return_value=response)
        _, files, source_fp = self.metadata.fetch_cocina(
            self.store.start_run("manifest.csv"), DRUID
        )
        self.assertNotEqual(self.source_fp, source_fp)
        self.assertEqual("changed", files[0]["sha1"])
        self.assertEqual('W/"new"', self.store.object_row(DRUID)["source_etag"])
        response.json.assert_called_once()

    def test_corrupt_cache_forces_unconditional_get(self):
        self.seed_validators(fresh=True)
        self.cache.write_text("corrupt")
        response = self.response(200, data=self.data, etag='W/"repaired"')
        self.http.get = Mock(return_value=response)
        self.metadata.fetch_cocina(self.store.start_run("manifest.csv"), DRUID)
        self.assertEqual({}, self.http.get.call_args.kwargs["headers"])
        self.assertEqual(self.data, json.loads(self.cache.read_text()))

    def test_traject_output_with_unicode_line_separator_is_one_json_record(self):
        version_dir = self.root / "version"
        version_dir.mkdir()
        (version_dir / "cocina.json").write_text(json.dumps(self.data))
        metadata = {"id": [DRUID], "abstract_tesi": ["before\u2028after"]}
        result = Mock(
            returncode=0,
            stdout=json.dumps(metadata, ensure_ascii=False),
            stderr="",
        )

        with patch("sdr_harvest.metadata.subprocess.run", return_value=result):
            output = self.metadata.create_search_metadata(DRUID, version_dir)

        self.assertEqual(metadata, json.loads(output.read_text()))
        self.assertFalse((version_dir / "traject-input").exists())

    def test_publish_batches_documents_without_a_hard_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = []
            for druid in (DRUID, "bb005wc0080"):
                version_dir = root / druid
                version_dir.mkdir()
                document = {
                    "id": druid,
                    "child_count_i": 1,
                    "_childDocuments_": [{"id": f"{druid}-child"}],
                }
                (version_dir / "solr.json").write_text(json.dumps(document))
                items.append(
                    PublicationItem(
                        druid,
                        f"source-{druid}",
                        f"document-{druid}",
                        version_dir,
                    )
                )
            http = Mock()
            publisher = SolrPublisher(Settings(root, root, verify_tls=False), http)
            self.assertFalse(http.verify)
            update = Mock(status_code=200)
            update.json.return_value = {"responseHeader": {"status": 0}}
            http.post = Mock(return_value=update)
            receipts = publisher.publish_batch(items)
            request = http.post.call_args.kwargs
            payload = request["data"].decode()
            self.assertEqual(2, payload.count('"delete":'))
            self.assertEqual(2, payload.count('"add":'))
            self.assertNotIn('"commit":', payload)
            self.assertEqual(COMMIT_WITHIN_MS, request["params"]["commitWithin"])
            self.assertEqual(set(receipts), {item.druid for item in items})
            self.assertTrue(all(receipt.exists() for receipt in receipts.values()))
            http.get.assert_not_called()

    def test_corpus_publication_is_tracked_independently_per_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(f"identifier\n{DRUID}\n")
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            store.set_source(DRUID, "source", "1", [])
            version_dir = root / "versions" / DRUID / "source"
            version_dir.mkdir(parents=True)
            document = version_dir / "solr.json"
            document.write_text(json.dumps({"id": DRUID}))
            store.adopt_stage(
                DRUID,
                "document",
                "input",
                "output",
                SIGNATURES["document"],
                Path("/old-machine/.sdr-harvest/versions")
                / DRUID
                / "source"
                / "solr.json",
            )
            store.mark_built(DRUID, str(version_dir))

            receipt = version_dir / "receipt.json"
            receipt.write_text("{}")

            def publish_items(items):
                return {item.druid: receipt for item in items}

            with patch.object(
                SolrPublisher, "publish_batch", side_effect=publish_items
            ) as publish:
                staging_settings = Settings(
                    root, root, solr_url="https://staging.example/solr/core"
                )
                staging = CorpusPublisher(staging_settings, store)
                staging_batch = SolrPublisher(staging_settings, Mock()).publish_batch
                first = staging.publish(manifest, staging_batch, show_progress=False)
                second = staging.publish(manifest, staging_batch, show_progress=False)
                store.adopt_stage(
                    DRUID,
                    "document",
                    "input",
                    "changed-output",
                    SIGNATURES["document"],
                    document,
                )
                changed = staging.publish(manifest, staging_batch, show_progress=False)
                production_settings = Settings(
                    root, root, solr_url="https://production.example/solr/core"
                )
                production = CorpusPublisher(production_settings, store)
                production_batch = SolrPublisher(
                    production_settings, Mock()
                ).publish_batch
                third = production.publish(
                    manifest, production_batch, show_progress=False
                )

            self.assertEqual(1, first["published"])
            self.assertEqual(1, second["skipped"])
            self.assertEqual(1, changed["published"])
            self.assertEqual(1, third["published"])
            self.assertEqual(3, publish.call_count)
            store.close()

    def test_document_error_splits_batch_and_preserves_per_object_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            druids = (DRUID, "bb005wc0080")
            manifest = root / "manifest.csv"
            manifest.write_text("identifier\n" + "\n".join(druids) + "\n")
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest(set(druids))
            for druid in druids:
                store.set_source(druid, f"source-{druid}", "1", [])
                version_dir = root / "versions" / druid / f"source-{druid}"
                version_dir.mkdir(parents=True)
                document = version_dir / "solr.json"
                document.write_text(json.dumps({"id": druid, "child_count_i": 0}))
                store.adopt_stage(
                    druid,
                    "document",
                    "input",
                    "output",
                    SIGNATURES["document"],
                    document,
                )

            batch_sizes = []

            def publish_items(items):
                batch_sizes.append(len(items))
                if len(items) > 1:
                    raise BatchPublishError("bad document", splittable=True)
                receipt = items[0].version_dir / "receipt.json"
                receipt.write_text("{}")
                return {items[0].druid: receipt}

            summary = CorpusPublisher(Settings(root, root), store).publish(
                manifest, publish_items, show_progress=False
            )

            self.assertEqual([2, 1, 1], batch_sizes)
            self.assertEqual(2, summary["published"])
            self.assertEqual(0, summary["failed"])
            for druid in druids:
                publication = store.db.execute(
                    "SELECT status,attempt_count FROM publications WHERE druid=?",
                    (druid,),
                ).fetchone()
                self.assertEqual(("succeeded", 2), tuple(publication))
            store.close()

    def test_deterministic_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            attempts = StageAttempts(Settings(root, root, max_retries=5), store)
            operation = Mock(side_effect=ValueError("bad data"))
            with self.assertRaises(ValueError):
                attempts.run(
                    store.start_run("manifest.csv"),
                    DRUID,
                    "download",
                    "input",
                    "signature",
                    operation,
                )
            self.assertEqual(1, operation.call_count)
            self.assertEqual("failed", store.stage(DRUID, "download").status)
            store.close()
