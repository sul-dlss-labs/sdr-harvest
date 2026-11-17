import fcntl
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from sdr_harvest.cli import (
    main,
)
from sdr_harvest.core import (
    SIGNATURES,
    Settings,
)
from sdr_harvest.dark import read_dark_list
from sdr_harvest.manifests import (
    merge_manifests,
    parse_manifest,
)
from sdr_harvest.publisher import (
    COMMIT_WITHIN_MS,
    CorpusPublisher,
)
from sdr_harvest.state import StateStore
from tests.support import DRUID


class ManifestTest(unittest.TestCase):
    def test_reads_headered_csv_and_normalizes_druid_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(f"identifier,title\ndruid:{DRUID},Example\n")
            self.assertEqual({DRUID}, parse_manifest(path))

    def test_rejects_manifest_without_druids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text("identifier\nnot-an-id\n")
            with self.assertRaises(ValueError):
                parse_manifest(path)

    def test_merges_sorts_and_deduplicates_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            output = root / "manifest.csv"
            first.write_text(f"Druid\n{DRUID}\nbb005wc0080\n")
            second.write_text(f"identifier\ndruid:{DRUID}\ncc006xy0091\n")
            summary = merge_manifests([first, second], output)
            self.assertEqual(3, summary["unique_records"])
            self.assertEqual(1, summary["duplicates_removed"])
            self.assertEqual(
                ["identifier", DRUID, "bb005wc0080", "cc006xy0091"],
                output.read_text().splitlines(),
            )

    def test_merge_refuses_to_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            first.write_text(f"Druid\n{DRUID}\n")
            second.write_text("Druid\nbb005wc0080\n")
            with self.assertRaises(ValueError):
                merge_manifests([first, second], first)


class DarkTest(unittest.TestCase):
    OTHER = "bb005wc0080"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state_dir = self.root / "state"
        self.manifest = self.root / "manifest.csv"
        self.manifest.write_text(f"identifier\ndruid:{DRUID}\n{self.OTHER}\n")
        self.dark_list = self.root / "dark.csv"
        store = StateStore(self.state_dir / "state.sqlite3")
        store.reconcile_manifest({DRUID, self.OTHER})
        for druid in (DRUID, self.OTHER):
            store.set_source(druid, "source", "1", [])
            store.adopt_stage(
                druid,
                "document",
                "in",
                "out",
                SIGNATURES["document"],
                self.state_dir / "versions" / druid / "source" / "solr.json",
            )
            run_id = store.start_run(str(self.manifest))
            store.begin_attempt(run_id, druid, "cocina", "in", "sig", "log")
            store.begin_publication(druid, "https://solr.example/core", "source", "out")
            for path in (
                self.state_dir / "sources" / druid / "cocina.json",
                self.state_dir / "versions" / druid / "source" / "solr.json",
                self.state_dir / "logs" / "1" / druid / "cocina.jsonl",
                self.state_dir / "logs" / "publish" / "abc" / f"{druid}.jsonl",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
        store.close()

    def tearDown(self):
        self.temp.cleanup()

    def _darken(self, *arguments):
        http = Mock()
        http.post.return_value = Mock(
            status_code=200, json=Mock(return_value={"responseHeader": {"status": 0}})
        )
        session = Mock()
        session.__enter__ = Mock(return_value=http)
        session.__exit__ = Mock(return_value=False)
        stdout = io.StringIO()
        with (
            patch("sdr_harvest.cli.requests.Session", return_value=session),
            redirect_stdout(stdout),
        ):
            main(
                [
                    "--state-dir",
                    str(self.state_dir),
                    "--dark-list",
                    str(self.dark_list),
                    "darken",
                    "--manifest",
                    str(self.manifest),
                    *arguments,
                ]
            )
        return json.loads(stdout.getvalue()), http

    def test_darken_records_object_and_purges_manifest_state_and_solr(self):
        summary, http = self._darken(
            "--druid",
            f"druid:{DRUID.upper()}",
            "--target",
            "https://solr.example/core/",
        )

        self.assertEqual([DRUID], summary["added"])
        self.assertEqual(1, summary["manifest_rows_removed"])
        self.assertEqual([DRUID], summary["local_state_purged"])
        self.assertEqual(["https://solr.example/core"], summary["solr_targets"])
        self.assertEqual(
            ["druid,darkened_on", f"{DRUID},{datetime.now().date().isoformat()}"],
            self.dark_list.read_text().splitlines(),
        )
        self.assertEqual(
            ["identifier", self.OTHER], self.manifest.read_text().splitlines()
        )
        self.assertEqual([], list(self.state_dir.rglob(f"*{DRUID}*")))
        db = sqlite3.connect(self.state_dir / "state.sqlite3")
        for table in (
            "objects",
            "source_files",
            "stage_state",
            "attempts",
            "publications",
        ):
            self.assertEqual(
                0,
                db.execute(
                    f"SELECT count(*) FROM {table} WHERE druid=?", (DRUID,)
                ).fetchone()[0],
                table,
            )
        self.assertEqual(
            1,
            db.execute(
                "SELECT count(*) FROM objects WHERE druid=?", (self.OTHER,)
            ).fetchone()[0],
        )
        self.assertEqual(
            1,
            db.execute(
                "SELECT count(*) FROM attempts WHERE druid=?", (self.OTHER,)
            ).fetchone()[0],
        )
        db.close()
        self.assertTrue((self.state_dir / "versions" / self.OTHER).exists())
        request = http.post.call_args
        self.assertEqual("https://solr.example/core/update", request.args[0])
        self.assertEqual(
            f'{{"delete":{{"query":"_root_:\\"{DRUID}\\""}}}}',
            request.kwargs["data"].decode(),
        )
        self.assertEqual(COMMIT_WITHIN_MS, request.kwargs["params"]["commitWithin"])

    def test_darken_without_druid_reapplies_list_and_keeps_original_date(self):
        self.dark_list.write_text(f"druid,darkened_on\n{DRUID},2026-01-01\n")

        summary, http = self._darken("--target", "https://other.example/core")
        again, _ = self._darken()

        self.assertEqual([], summary["added"])
        self.assertEqual([DRUID], summary["local_state_purged"])
        self.assertEqual(1, http.post.call_count)
        self.assertEqual(0, again["manifest_rows_removed"])
        self.assertEqual([], again["local_state_purged"])
        self.assertEqual(
            f"druid,darkened_on\n{DRUID},2026-01-01\n", self.dark_list.read_text()
        )

    def test_darken_changes_nothing_while_a_run_holds_the_lock(self):
        with (self.state_dir / "run.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                self._darken("--druid", DRUID)

        self.assertIn("already active", stderr.getvalue())
        self.assertFalse(self.dark_list.exists())
        self.assertIn(DRUID, self.manifest.read_text())

    def test_darken_rejects_invalid_druid(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(
                [
                    "--state-dir",
                    str(self.state_dir),
                    "--dark-list",
                    str(self.dark_list),
                    "darken",
                    "--druid",
                    "not-a-druid",
                ]
            )
        self.assertIn("Invalid DRUID", stderr.getvalue())
        self.assertFalse(self.dark_list.exists())

    def test_missing_dark_list_means_nothing_is_dark(self):
        self.assertEqual(set(), read_dark_list(self.dark_list))
        self.assertEqual(set(), read_dark_list(None))

    def test_merge_excludes_dark_objects(self):
        second = self.root / "second.csv"
        second.write_text(f"identifier\n{DRUID}\ncc006xy0091\n")
        output = self.root / "merged.csv"

        summary = merge_manifests([self.manifest, second], output, {DRUID})

        self.assertEqual(
            ["identifier", self.OTHER, "cc006xy0091"], output.read_text().splitlines()
        )
        self.assertEqual(1, summary["dark_excluded"])
        self.assertEqual(1, summary["duplicates_removed"])

    def test_publish_deletes_dark_objects_from_its_target(self):
        self.dark_list.write_text(f"druid,darkened_on\n{DRUID},2026-01-01\n")
        http = Mock()
        http.post.return_value = Mock(
            status_code=200, json=Mock(return_value={"responseHeader": {"status": 0}})
        )
        session = Mock()
        session.__enter__ = Mock(return_value=http)
        session.__exit__ = Mock(return_value=False)
        stdout = io.StringIO()
        with (
            patch("sdr_harvest.cli.requests.Session", return_value=session),
            patch.object(
                CorpusPublisher, "publish", return_value={"failed": 0, "not_ready": 0}
            ),
            redirect_stdout(stdout),
        ):
            main(
                [
                    "--state-dir",
                    str(self.state_dir),
                    "--dark-list",
                    str(self.dark_list),
                    "publish",
                    "--manifest",
                    str(self.manifest),
                    "--target",
                    "https://solr.example/core",
                    "--no-progress",
                ]
            )

        self.assertEqual(1, json.loads(stdout.getvalue())["dark_deleted"])
        self.assertIn(
            f'_root_:\\"{DRUID}\\"', http.post.call_args.kwargs["data"].decode()
        )

    def test_corpus_publisher_excludes_dark_objects(self):
        self.dark_list.write_text(f"druid,darkened_on\n{DRUID},2026-01-01\n")
        store = StateStore(self.state_dir / "state.sqlite3")
        settings = Settings(
            self.root,
            self.state_dir,
            solr_url="https://solr.example/core",
            dark_list=self.dark_list,
        )

        summary = CorpusPublisher(settings, store).publish(
            self.manifest,
            lambda items: {item.druid: item.document_path for item in items},
            show_progress=False,
        )
        store.close()

        self.assertEqual(1, summary["total"])
