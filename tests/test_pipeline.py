from contextlib import redirect_stderr, redirect_stdout

import fcntl
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf
import pymupdf4llm

from sdr_harvest.cli import (
    RESOURCE_TRACKER_WARNING_FILTER,
    _configure_child_warning_filters,
    _settings,
    main,
    parser,
)
from sdr_harvest.attempts import StageAttempts
from sdr_harvest.chunk import CHUNK_OVERLAP, CHUNK_SIZE, Chunker
from sdr_harvest.core import (
    ALTO_EXTRACT_SIGNATURE,
    PDF_EXTRACT_SIGNATURE,
    SIGNATURES,
    SOURCE_INVENTORY_SIGNATURE,
    Settings,
    StageError,
    TransientStageError,
    directory_fingerprint,
    file_sha256,
    interruptible_thread_pool,
)
from sdr_harvest.create_solr_document import SolrDocumentBuilder
from sdr_harvest.download import FileDownloader
from sdr_harvest.embed import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CONCURRENCY,
    ERROR_BODY_LIMIT,
    Embedder,
    http_error_message,
    retry_after_seconds,
    retrieval_document,
    retrieval_title,
)
from sdr_harvest.extract_pdf import PdfExtractionStrategy, extract_pdf_to_markdown
from sdr_harvest.extract_text import REUSE_MARKER, TextExtractor
from sdr_harvest.dark import read_dark_list
from sdr_harvest.manifests import (
    cocina_pdf_files,
    cocina_page_numbers,
    cocina_source_files,
    merge_manifests,
    parse_manifest,
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
from sdr_harvest.state import STAGES, StateStore


DRUID = "ab123cd4567"


def cocina(version=1, digest="abc"):
    return {
        "externalIdentifier": f"druid:{DRUID}",
        "version": version,
        "description": {"title": [{"value": "Example"}]},
        "structural": {
            "contains": [{
                "structural": {"contains": [{
                    "externalIdentifier": "file:1",
                    "filename": "example.pdf",
                    "hasMimeType": "application/pdf",
                    "size": 3,
                    "version": 1,
                    "hasMessageDigests": [{"type": "sha1", "digest": digest}],
                }]}
            }]
        },
    }


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
        with patch(
            "sdr_harvest.cli.requests.Session", return_value=session
        ), redirect_stdout(stdout):
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


class FingerprintTest(unittest.TestCase):
    def test_chunk_signature_records_current_splitter_parameters(self):
        self.assertEqual(
            f"recursive-{CHUNK_SIZE}-{CHUNK_OVERLAP}-v2",
            SIGNATURES["chunk"],
        )

    def test_embedding_signature_records_retrieval_format(self):
        self.assertEqual(
            "gemini-embedding-2-768-retrieval-document-v2",
            SIGNATURES["embed"],
        )

    def test_formats_embedding_text_for_asymmetric_retrieval(self):
        metadata = {"title_display_tesi": ["Interview with John Lynch"]}
        title = retrieval_title(metadata)

        self.assertEqual("Interview with John Lynch", title)
        self.assertEqual(
            "title: Interview with John Lynch | text: Bill Walsh reviewed the film.",
            retrieval_document(title, "Bill Walsh reviewed the film."),
        )
        self.assertEqual("none", retrieval_title({}))

    def test_extracts_pdf_identity_and_digests(self):
        files = cocina_pdf_files(cocina())
        self.assertEqual("example.pdf", files[0]["filename"])
        self.assertEqual("abc", files[0]["sha1"])

    def test_prefers_object_pdf_over_page_pdfs(self):
        data = cocina()
        data["structural"]["contains"] = [
            {
                "type": "https://cocina.sul.stanford.edu/models/resources/page",
                "structural": {
                    "contains": [
                        {
                            "externalIdentifier": "file:page-1",
                            "filename": "page-1.pdf",
                            "hasMimeType": "application/pdf",
                        }
                    ]
                },
            },
            {
                "type": "https://cocina.sul.stanford.edu/models/resources/object",
                "structural": {
                    "contains": [
                        {
                            "externalIdentifier": "file:full",
                            "filename": "full-document.pdf",
                            "hasMimeType": "application/pdf",
                        }
                    ]
                },
            },
        ]

        self.assertEqual(
            ["full-document.pdf"],
            [file["filename"] for file in cocina_pdf_files(data)],
        )

    def test_keeps_page_pdfs_when_no_object_pdf_exists(self):
        data = cocina()
        data["structural"]["contains"][0]["type"] = (
            "https://cocina.sul.stanford.edu/models/resources/page"
        )

        self.assertEqual(
            ["example.pdf"],
            [file["filename"] for file in cocina_pdf_files(data)],
        )

    def test_book_prefers_complete_page_alto_over_pdfs(self):
        data = cocina()
        data["type"] = "https://cocina.sul.stanford.edu/models/book"
        data["structural"]["contains"] = [
            self.page_resource(1, include_alto=True),
            self.page_resource(2, include_alto=True),
            self.full_pdf_resource(),
        ]

        self.assertEqual(
            ["page-1.xml", "page-2.xml"],
            [file["filename"] for file in cocina_source_files(data)],
        )
        self.assertEqual(
            {
                "page-1.pdf": "1",
                "page-1.xml": "1",
                "page-2.pdf": "2",
                "page-2.xml": "2",
            },
            cocina_page_numbers(data),
        )

    def test_book_falls_back_to_object_pdf_when_page_alto_is_incomplete(self):
        data = cocina()
        data["type"] = "https://cocina.sul.stanford.edu/models/book"
        data["structural"]["contains"] = [
            self.page_resource(1, include_alto=True),
            self.page_resource(2, include_alto=False),
            self.full_pdf_resource(),
        ]

        self.assertEqual(
            ["full.pdf"],
            [file["filename"] for file in cocina_source_files(data)],
        )

    @staticmethod
    def page_resource(number: int, *, include_alto: bool) -> dict:
        files = [
            {
                "externalIdentifier": f"file:page-pdf-{number}",
                "filename": f"page-{number}.pdf",
                "hasMimeType": "application/pdf",
            }
        ]
        if include_alto:
            files.append(
                {
                    "externalIdentifier": f"file:page-alto-{number}",
                    "filename": f"page-{number}.xml",
                    "hasMimeType": "application/xml",
                    "use": "transcription",
                }
            )
        return {
            "externalIdentifier": f"resource:page-{number}",
            "type": "https://cocina.sul.stanford.edu/models/resources/page",
            "structural": {"contains": files},
        }

    @staticmethod
    def full_pdf_resource() -> dict:
        return {
            "type": "https://cocina.sul.stanford.edu/models/resources/object",
            "structural": {
                "contains": [
                    {
                        "externalIdentifier": "file:full",
                        "filename": "full.pdf",
                        "hasMimeType": "application/pdf",
                    }
                ]
            },
        }

    def test_metadata_and_file_changes_alter_source_fingerprint(self):
        first = cocina()
        changed_metadata = cocina()
        changed_metadata["description"]["title"][0]["value"] = "Changed"
        changed_file = cocina(digest="def")
        self.assertNotEqual(source_fingerprint(first, cocina_pdf_files(first)), source_fingerprint(changed_metadata, cocina_pdf_files(changed_metadata)))
        self.assertNotEqual(source_fingerprint(first, cocina_pdf_files(first)), source_fingerprint(changed_file, cocina_pdf_files(changed_file)))


class TextExtractorTest(unittest.TestCase):
    def test_uses_strategy_specific_signatures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pdfs"
            source.mkdir()
            cocina_path = root / "cocina.json"
            extractor = TextExtractor()

            cocina_path.write_text(
                json.dumps({"type": "https://cocina.sul.stanford.edu/models/book"})
            )
            source.joinpath("page.xml").write_text("<alto/>")
            self.assertEqual(ALTO_EXTRACT_SIGNATURE, extractor.signature(root))

            source.joinpath("page.xml").unlink()
            source.joinpath("document.pdf").write_bytes(b"pdf")
            cocina_path.write_text(
                json.dumps(
                    {"type": "https://cocina.sul.stanford.edu/models/document"}
                )
            )
            self.assertEqual(PDF_EXTRACT_SIGNATURE, extractor.signature(root))

    def test_book_extracts_alto_and_joins_line_break_hyphenation(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("cocina.json").write_text(
                json.dumps(
                    {
                        "type": "https://cocina.sul.stanford.edu/models/book",
                        "structural": {
                            "contains": [
                                {
                                    "type": "https://cocina.sul.stanford.edu/"
                                    "models/resources/page",
                                    "structural": {
                                        "contains": [
                                            {
                                                "filename": "page-1.xml",
                                                "hasMimeType": "application/xml",
                                                "use": "transcription",
                                            }
                                        ]
                                    },
                                }
                            ]
                        },
                    }
                )
            )
            source = version_dir / "pdfs"
            source.mkdir()
            source.joinpath("page-1.xml").write_text(
                """<?xml version="1.0"?>
                <alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
                  <Layout><Page><PrintSpace><TextBlock>
                    <TextLine><String CONTENT="bac"/><HYP CONTENT="-"/></TextLine>
                    <TextLine><String CONTENT="teria"/><SP/><String CONTENT="cause"/></TextLine>
                    <TextLine><String CONTENT="disease."/></TextLine>
                  </TextBlock></PrintSpace></Page></Layout>
                </alto>"""
            )

            output = TextExtractor().run(version_dir)

            self.assertEqual(
                "bacteria cause\ndisease.",
                output.joinpath("page-1.md").read_text(),
            )
            self.assertEqual(
                {
                    "page-1.md": {
                        "page": "1",
                        "source_file": "page-1.xml",
                    }
                },
                json.loads(output.joinpath("pages.json").read_text()),
            )

    def test_non_book_uses_pdf_strategy(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("cocina.json").write_text(
                json.dumps(
                    {"type": "https://cocina.sul.stanford.edu/models/document"}
                )
            )
            source = version_dir / "pdfs"
            source.mkdir()
            source.joinpath("document.pdf").write_bytes(b"pdf")

            with patch(
                "sdr_harvest.extract_pdf.pymupdf4llm.to_markdown",
                return_value="PDF text",
            ) as to_markdown:
                output = TextExtractor().run(version_dir)

            self.assertEqual("PDF text", output.joinpath("document.md").read_text())
            to_markdown.assert_called_once()

    def test_pdf_pages_are_extracted_and_chunked_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("cocina.json").write_text(
                json.dumps(
                    {"type": "https://cocina.sul.stanford.edu/models/document"}
                )
            )
            version_dir.joinpath("metadata.json").write_text("{}")
            source = version_dir / "pdfs"
            source.mkdir()
            source.joinpath("document.pdf").write_bytes(b"pdf")
            first_page = "first " * 250
            second_page = "second " * 250
            page_chunks = [
                {
                    "text": first_page,
                    "metadata": {"page_number": 1},
                },
                {
                    "text": second_page,
                    "metadata": {"page_number": 2},
                },
            ]

            with patch(
                "sdr_harvest.extract_pdf.pymupdf4llm.to_markdown",
                return_value=page_chunks,
            ):
                TextExtractor().run(version_dir)
            rows = pq.read_table(Chunker().run(DRUID, version_dir)).to_pylist()
            text_rows = [row for row in rows if row["page"] is not None]

            self.assertTrue(all(row["file"] == "document.md" for row in text_rows))
            self.assertEqual({"1", "2"}, {row["page"] for row in text_rows})
            self.assertTrue(
                all(row["source_file"] == "document.pdf" for row in text_rows)
            )
            self.assertEqual(
                first_page + second_page,
                (version_dir / "markdown" / "document.md").read_text(),
            )


class PdfExtractionRegressionTest(unittest.TestCase):
    def test_submits_pdf_extraction_to_the_shared_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "document.pdf"
            pdf.write_bytes(b"pdf")
            output = root / "markdown"
            output.mkdir()
            executor = Mock()

            def submit(operation, *args):
                result = operation(*args)
                future = Mock()
                future.result.return_value = result
                return future

            executor.submit.side_effect = submit
            with patch(
                "sdr_harvest.extract_pdf.pymupdf4llm.to_markdown",
                return_value="Extracted in a worker process",
            ):
                PdfExtractionStrategy(executor).extract([pdf], output)

            executor.submit.assert_called_once_with(
                extract_pdf_to_markdown, pdf, output / "document.md", None
            )
            self.assertEqual(
                "Extracted in a worker process",
                output.joinpath("document.md").read_text(),
            )

    def test_preserves_an_invisible_pdf_text_layer(self):
        """Regression test for hz508gx6219 losing all non-heading text."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "invisible-text-layer.pdf"
            document = pymupdf.open()
            page = document.new_page()
            phrase = "Invisible text layers must remain searchable. "
            page.insert_textbox(
                pymupdf.Rect(50, 50, 550, 750),
                phrase * 30,
                fontsize=11,
                render_mode=3,
            )
            document.save(pdf)
            document.close()

            with pymupdf.open(pdf) as check:
                self.assertGreater(len(check[0].get_text()), 1_000)

            output = root / "markdown"
            output.mkdir()
            PdfExtractionStrategy().extract([pdf], output)

            extracted = output.joinpath("invisible-text-layer.md").read_text()
            self.assertIn("Invisible text layers must remain searchable", extracted)

    def test_page_metadata_does_not_change_extracted_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "document.pdf"
            document = pymupdf.open()
            for page_number in range(1, 3):
                page = document.new_page()
                page.insert_text((72, 72), f"Content on page {page_number}")
            document.save(pdf)
            document.close()
            expected = pymupdf4llm.to_markdown(
                str(pdf),
                write_images=False,
                use_ocr=pymupdf4llm.ocr.OCRMode.NEVER,
            )
            output = root / "markdown"
            output.mkdir()

            PdfExtractionStrategy().extract([pdf], output)

            self.assertEqual(expected, output.joinpath("document.md").read_text())


class StreamingResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class FileDownloaderTest(unittest.TestCase):
    def file_info(self, content: bytes) -> dict:
        return {
            "filename": "example.pdf",
            "size": len(content),
            "sha1": hashlib.sha1(content).hexdigest(),
            "md5": hashlib.md5(content).hexdigest(),
        }

    def test_streams_and_validates_download(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            content = b"pdf content"
            http = Mock()
            http.get.return_value = StreamingResponse(content)

            output = FileDownloader(http).run(
                DRUID, [self.file_info(content)], version_dir
            )

            self.assertEqual(content, output.joinpath("example.pdf").read_bytes())
            http.get.assert_called_once_with(
                f"https://stacks.stanford.edu/file/{DRUID}/example.pdf",
                timeout=120,
                stream=True,
            )

    def test_mismatch_reports_all_expected_and_actual_values(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            expected = b"expected"
            actual = b"different content"
            http = Mock()
            http.get.return_value = StreamingResponse(actual)

            with self.assertRaisesRegex(
                StageError, "Download integrity mismatch"
            ) as raised:
                FileDownloader(http).run(
                    DRUID, [self.file_info(expected)], version_dir
                )

            message = str(raised.exception)
            self.assertIn(
                f"size expected={len(expected)} actual={len(actual)}", message
            )
            self.assertIn(
                f"SHA-1 expected={hashlib.sha1(expected).hexdigest()} "
                f"actual={hashlib.sha1(actual).hexdigest()}",
                message,
            )
            self.assertIn(
                f"MD5 expected={hashlib.md5(expected).hexdigest()} "
                f"actual={hashlib.md5(actual).hexdigest()}",
                message,
            )
            self.assertIn("invalid_file=deleted", message)
            self.assertFalse(
                version_dir.joinpath("pdfs", "example.pdf.tmp").exists()
            )
            self.assertFalse(
                version_dir.joinpath("pdfs", "example.pdf.invalid").exists()
            )

    def test_can_retain_invalid_download(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            expected = b"expected"
            actual = b"different content"
            http = Mock()
            http.get.return_value = StreamingResponse(actual)

            with self.assertRaises(StageError) as raised:
                FileDownloader(http, keep_failed_downloads=True).run(
                    DRUID, [self.file_info(expected)], version_dir
                )

            invalid = version_dir / "pdfs" / "example.pdf.invalid"
            self.assertEqual(actual, invalid.read_bytes())
            self.assertIn(f"invalid_file={invalid}", str(raised.exception))


class ChunkerTest(unittest.TestCase):
    def test_uses_larger_overlapping_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text(
                json.dumps({"title_tesi": "Example"})
            )
            markdown = version_dir / "markdown"
            markdown.mkdir()
            markdown.joinpath("example.md").write_text(
                " ".join(f"word-{index:04d}" for index in range(600))
            )

            table = Chunker().run(DRUID, version_dir)
            rows = pq.read_table(table).to_pylist()
            text_rows = [row for row in rows if row["file"] == "example.md"]

            self.assertGreater(len(text_rows), 1)
            self.assertTrue(
                all(len(row["text"]) <= CHUNK_SIZE for row in text_rows)
            )
            self.assertTrue(
                all(len(row["text"]) > CHUNK_OVERLAP for row in text_rows)
            )
            self.assertGreater(max(len(row["text"]) for row in text_rows), 1_000)
            for first, second in zip(text_rows, text_rows[1:], strict=False):
                self.assertTrue(
                    set(first["text"].split()[-30:])
                    & set(second["text"].split()[:30])
                )

    def test_carries_extracted_page_metadata_into_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("metadata.json").write_text("{}")
            markdown = version_dir / "markdown"
            markdown.mkdir()
            markdown.joinpath("page.md").write_text("ALTO page text")
            markdown.joinpath("pages.json").write_text(
                json.dumps(
                    {
                        "page.md": {
                            "page": "42",
                            "source_file": "page.xml",
                        }
                    }
                )
            )

            rows = pq.read_table(Chunker().run(DRUID, version_dir)).to_pylist()
            row = next(row for row in rows if row["file"] == "page.md")

            self.assertEqual("42", row["page"])
            self.assertEqual("page.xml", row["source_file"])

    def test_leaves_page_unknown_without_page_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("metadata.json").write_text("{}")
            markdown = version_dir / "markdown"
            markdown.mkdir()
            markdown.joinpath("page-10.md").write_text("Text without a known page")

            rows = pq.read_table(Chunker().run(DRUID, version_dir)).to_pylist()
            row = next(row for row in rows if row["file"] == "page-10.md")

            self.assertIsNone(row["page"])

    def test_leaves_page_unknown_when_chunk_start_matches_no_page_range(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("metadata.json").write_text("{}")
            markdown = version_dir / "markdown"
            markdown.mkdir()
            markdown.joinpath("document.md").write_text("Unmatched text")
            markdown.joinpath("pages.json").write_text(
                json.dumps(
                    {
                        "document.md": {
                            "pages": [{"page": "9", "start": 100, "end": 200}],
                            "source_file": "document.pdf",
                        }
                    }
                )
            )

            rows = pq.read_table(Chunker().run(DRUID, version_dir)).to_pylist()
            row = next(row for row in rows if row["file"] == "document.md")

            self.assertIsNone(row["page"])


class SolrDocumentBuilderTest(unittest.TestCase):
    def test_records_chunk_pages_as_page_ss_on_child_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("metadata.json").write_text("{}")
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "file": "_metadata_",
                            "source_file": "_metadata_",
                            "page": None,
                            "chunk_index": 0,
                            "text": "Metadata",
                            "embedding": [0.1],
                        },
                        {
                            "file": "document.md",
                            "source_file": "document.pdf",
                            "page": "1",
                            "chunk_index": 0,
                            "text": "First page",
                            "embedding": [0.2],
                        },
                        {
                            "file": "document.md",
                            "source_file": "document.pdf",
                            "page": "2",
                            "chunk_index": 1,
                            "text": "Second page",
                            "embedding": [0.3],
                        },
                    ]
                ),
                version_dir / "embeddings.parquet",
            )

            output = SolrDocumentBuilder().run(DRUID, "source", version_dir)
            children = json.loads(output.read_text())["_childDocuments_"]

            self.assertNotIn("page_ss", children[0])
            self.assertEqual(["1", "2"], [child["page_ss"] for child in children[1:]])
            self.assertEqual(3, len({child["id"] for child in children}))


class EmbedderTest(unittest.TestCase):
    def test_uses_reduced_batch_size_and_concurrency(self):
        self.assertEqual(50, EMBEDDING_BATCH_SIZE)
        self.assertEqual(2, EMBEDDING_CONCURRENCY)

    def test_http_error_includes_body_and_rate_limit_headers(self):
        response = Mock(
            status_code=429,
            headers={
                "Retry-After": "30",
                "X-Request-ID": "request-123",
                "X-LiteLLM-Call-ID": "call-456",
            },
            text='{"error":{"message":"rate limit exceeded"}}',
        )

        self.assertEqual(
            "LiteLLM HTTP 429; Retry-After=30; X-Request-ID=request-123; "
            "X-LiteLLM-Call-ID=call-456; "
            'body={"error":{"message":"rate limit exceeded"}}',
            http_error_message(response),
        )

    def test_http_error_truncates_large_response_body(self):
        response = Mock(
            status_code=500,
            headers={},
            text="x" * (ERROR_BODY_LIMIT + 1),
        )

        message = http_error_message(response)

        self.assertIn("x" * ERROR_BODY_LIMIT, message)
        self.assertTrue(message.endswith("... [truncated]"))

    def test_parses_retry_after_seconds_and_http_date(self):
        response = Mock(headers={"Retry-After": "12.5"})
        self.assertEqual(12.5, retry_after_seconds(response))

        response.headers = {"Retry-After": "Tue, 18 Aug 2026 17:00:30 GMT"}
        self.assertEqual(
            30.0,
            retry_after_seconds(
                response, now=datetime(2026, 8, 18, 17, 0, tzinfo=UTC)
            ),
        )

    def test_parses_litellm_limit_reset_from_error_body(self):
        response = Mock(
            headers={},
            text=json.dumps(
                {
                    "error": {
                        "message": "Rate limit exceeded. Limit resets at: "
                        "2026-08-18 21:17:58 UTC"
                    }
                }
            ),
        )

        self.assertEqual(
            58.0,
            retry_after_seconds(
                response, now=datetime(2026, 8, 18, 21, 17, tzinfo=UTC)
            ),
        )

    def test_sends_retrieval_formatted_text_and_preserves_raw_chunk_text(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text(
                json.dumps({"title_display_tesi": ["Example title"]})
            )
            raw_texts = ["First raw chunk", "Second raw chunk"]
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "object_id": DRUID,
                            "file": "example.md",
                            "chunk_index": index,
                            "text": text,
                        }
                        for index, text in enumerate(raw_texts)
                    ]
                ),
                version_dir / "chunks.parquet",
            )
            response = Mock(
                status_code=200,
                **{
                    "json.return_value": {
                        "embeddings": [
                            {"values": [0.1] * 768},
                            {"values": [0.2] * 768},
                        ]
                    }
                },
            )

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch("sdr_harvest.embed.requests.post", return_value=response) as post,
            ):
                output = Embedder().run(version_dir)

            calls = post.call_args_list
            self.assertEqual(1, len(calls))
            self.assertEqual(
                "https://dlss-aigateway-prod.stanford.edu/gemini/v1beta/models/"
                "gemini-embedding-2:batchEmbedContents",
                calls[0].args[0],
            )
            self.assertEqual(
                {"key": "test-key"},
                calls[0].kwargs["params"],
            )
            self.assertEqual(
                {"Content-Type": "application/json"},
                calls[0].kwargs["headers"],
            )
            requests = calls[0].kwargs["json"]["requests"]
            self.assertEqual(
                [
                    "title: Example title | text: First raw chunk",
                    "title: Example title | text: Second raw chunk",
                ],
                [item["content"]["parts"][0]["text"] for item in requests],
            )
            for item in requests:
                self.assertEqual(
                    "models/gemini-embedding-2", item["model"]
                )
                self.assertEqual(768, item["outputDimensionality"])
            self.assertEqual(
                raw_texts,
                pq.read_table(output).column("text").to_pylist(),
            )
            vector_starts = [
                vector[0]
                for vector in pq.read_table(output)
                .column("embedding")
                .to_pylist()
            ]
            self.assertAlmostEqual(0.1, vector_starts[0])
            self.assertAlmostEqual(0.2, vector_starts[1])

    def test_rejects_wrong_embedding_count_for_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text("{}")
            pq.write_table(
                pa.Table.from_pylist([{"text": "One chunk"}]),
                version_dir / "chunks.parquet",
            )
            response = Mock(
                status_code=200,
                **{
                    "json.return_value": {
                        "embeddings": [
                            {"values": [0.1] * 768},
                            {"values": [0.2] * 768},
                        ]
                    }
                },
            )

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch("sdr_harvest.embed.requests.post", return_value=response),
                self.assertRaisesRegex(
                    StageError, "LiteLLM returned 2 embeddings for 1 inputs"
                ),
            ):
                Embedder().run(version_dir)

    def test_splits_missing_embeddings_into_bounded_batches(self):
        texts = [f"chunk-{index}" for index in range(EMBEDDING_BATCH_SIZE * 2 + 1)]
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text("{}")
            pq.write_table(
                pa.Table.from_pylist([{"text": text} for text in texts]),
                version_dir / "chunks.parquet",
            )

            def embed_batch(batch, _key):
                return [[0.1] * 768 for _text in batch]

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch.object(
                    Embedder, "embed_batch", side_effect=embed_batch
                ) as embed_batch_call,
            ):
                Embedder().run(version_dir)

            self.assertCountEqual(
                [EMBEDDING_BATCH_SIZE, EMBEDDING_BATCH_SIZE, 1],
                [len(call.args[0]) for call in embed_batch_call.call_args_list],
            )

    def test_limits_embedding_concurrency_across_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("first", "second")]
            for root in roots:
                root.mkdir()
                (root / "metadata.json").write_text("{}")
                pq.write_table(
                    pa.Table.from_pylist(
                        [{"text": f"chunk-{index}"} for index in range(8)]
                    ),
                    root / "chunks.parquet",
                )

            lock = threading.Lock()
            active = 0
            maximum_active = 0

            def post(*_args, **_kwargs):
                nonlocal active, maximum_active
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.01)
                with lock:
                    active -= 1
                return Mock(
                    status_code=200,
                    **{
                        "json.return_value": {
                            "embeddings": [
                                {"values": [0.1] * 768}
                                for _text in _kwargs["json"]["requests"]
                            ]
                        }
                    },
                )

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch("sdr_harvest.embed.requests.post", side_effect=post),
                patch("sdr_harvest.embed.EMBEDDING_BATCH_SIZE", 2),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                list(executor.map(Embedder().run, roots))

            self.assertEqual(EMBEDDING_CONCURRENCY, maximum_active)

    def test_resumes_from_completed_embedding_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text("{}")
            pq.write_table(
                pa.Table.from_pylist(
                    [{"text": f"chunk-{index}"} for index in range(3)]
                ),
                version_dir / "chunks.parquet",
            )

            def fail_last(texts, _key):
                index = int(texts[0].rsplit("-", 1)[-1])
                if index == 2:
                    time.sleep(0.03)
                    raise TransientStageError("try again")
                return [[float(index)] * 768]

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch("sdr_harvest.embed.EMBEDDING_BATCH_SIZE", 1),
                patch.object(Embedder, "embed_batch", side_effect=fail_last),
                self.assertRaises(TransientStageError),
            ):
                Embedder().run(version_dir)

            def complete(texts, _key):
                index = int(texts[0].rsplit("-", 1)[-1])
                return [[float(index)] * 768]

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch("sdr_harvest.embed.EMBEDDING_BATCH_SIZE", 1),
                patch.object(
                    Embedder, "embed_batch", side_effect=complete
                ) as embed_batch,
            ):
                output = Embedder().run(version_dir)

            embed_batch.assert_called_once()
            self.assertIn("chunk-2", embed_batch.call_args.args[0][0])
            vectors = pq.read_table(output).column("embedding").to_pylist()
            self.assertEqual([0.0, 1.0, 2.0], [vector[0] for vector in vectors])

    def test_discards_checkpoint_when_embedding_inputs_change(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text("{}")

            def write_chunks(second_text):
                pq.write_table(
                    pa.Table.from_pylist(
                        [{"text": "first"}, {"text": second_text}]
                    ),
                    version_dir / "chunks.parquet",
                )

            write_chunks("second")
            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch.object(
                    Embedder,
                    "embed_batch",
                    return_value=[[0.1] * 768, [0.1] * 768],
                ),
            ):
                Embedder().run(version_dir)

            write_chunks("changed")
            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch.object(
                    Embedder,
                    "embed_batch",
                    return_value=[[0.2] * 768, [0.2] * 768],
                ) as embed_batch,
            ):
                Embedder().run(version_dir)

            embed_batch.assert_called_once()
            self.assertEqual(2, len(embed_batch.call_args.args[0]))

    def test_requires_litellm_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(StageError, "LITELLM_API_KEY is not set"):
                    Embedder().run(Path(directory))


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
        self.assertTrue(all(self.store.stage(DRUID, stage).status == "pending" for stage in STAGES[1:]))

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
        self.store.adopt_stage(
            DRUID, "document", "in", "out", "sig", artifact
        )
        self.store.mark_built(DRUID, str(artifact.parent))

        stored = self.store.db.execute(
            "SELECT artifact_path FROM stage_state WHERE druid=? AND stage='document'",
            (DRUID,),
        ).fetchone()[0]
        current = self.store.object_row(DRUID)["current_artifact_dir"]

        self.assertEqual(
            str(Path("versions") / DRUID / "source" / "solr.json"), stored
        )
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
            row[1]
            for row in migrated.db.execute("PRAGMA table_info(publications)")
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
            attempts = store.db.execute("SELECT status,transient FROM attempts ORDER BY id").fetchall()
            self.assertEqual([("failed", 1), ("succeeded", None)], [tuple(row) for row in attempts])
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
                attempts.run(
                    run_id, DRUID, "download", "input", "signature", operation
                )

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


class InterruptTest(unittest.TestCase):
    def test_resource_tracker_filter_is_targeted_preserved_and_idempotent(self):
        existing = "default::DeprecationWarning"
        with patch.dict(os.environ, {"PYTHONWARNINGS": existing}):
            _configure_child_warning_filters()
            _configure_child_warning_filters()
            filters = os.environ["PYTHONWARNINGS"].split(",")
            self.assertIn(existing, filters)
            self.assertEqual(1, filters.count(RESOURCE_TRACKER_WARNING_FILTER))

    def test_thread_pool_does_not_wait_after_keyboard_interrupt(self):
        executor = Mock(spec=ThreadPoolExecutor)
        with patch(
            "sdr_harvest.pipeline.concurrent.futures.ThreadPoolExecutor",
            return_value=executor,
        ):
            with self.assertRaises(KeyboardInterrupt):
                with interruptible_thread_pool(4):
                    raise KeyboardInterrupt
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_cli_exits_immediately_with_status_130(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(f"identifier\n{DRUID}\n")
            with (
                patch("sdr_harvest.cli.Pipeline.run", side_effect=KeyboardInterrupt),
                patch("sdr_harvest.cli.os._exit", side_effect=SystemExit(130)) as exit_,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "--state-dir",
                        str(root / "state"),
                        "run",
                        "--manifest",
                        str(manifest),
                        "--no-progress",
                    ]
                )
            self.assertEqual(130, raised.exception.code)
            exit_.assert_called_once_with(130)


class CliErrorTest(unittest.TestCase):
    def test_build_can_retain_failed_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser().parse_args(
                [
                    "run",
                    "--manifest",
                    "manifest.csv",
                    "--keep-failed-downloads",
                ]
            )
            args.state_dir = root / "state"

            self.assertTrue(_settings(root, args).keep_failed_downloads)

    def test_publish_insecure_disables_tls_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser().parse_args(
                [
                    "publish",
                    "--manifest",
                    "manifest.csv",
                    "--target",
                    "https://localhost:8983/solr/collection",
                    "--insecure",
                ]
            )
            args.state_dir = root / "state"

            self.assertFalse(_settings(root, args).verify_tls)

    def test_stage_error_is_printed_without_an_exception_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(f"identifier\n{DRUID}\n")
            stderr = io.StringIO()
            with (
                patch(
                    "sdr_harvest.cli.Pipeline.run",
                    side_effect=StageError("Another sdr-harvest run is already active"),
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "--state-dir",
                        str(root / "state"),
                        "run",
                        "--manifest",
                        str(manifest),
                        "--no-progress",
                    ]
                )

            self.assertEqual(1, raised.exception.code)
            self.assertEqual(
                "Error: Another sdr-harvest run is already active\n",
                stderr.getvalue(),
            )


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
            store.adopt_stage(
                DRUID, stage, input_fp, output_fp, signature, artifact
            )
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

            estimate, pending = Pipeline(
                Settings(root, root), store
            )._plan_work([DRUID], show_progress=False)

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
                store.adopt_stage(DRUID, stage, input_fp, output_fp, SIGNATURES[stage], artifact)
                input_fp = output_fp
            progress_events = []
            pipeline = Pipeline(
                Settings(root, root),
                store,
                progress_callback=lambda druid, stage, event: progress_events.append(
                    (druid, stage, event)
                ),
            )
            with (
                patch("sdr_harvest.pipeline.StageAttempts") as attempts_class,
                patch("sdr_harvest.pipeline.MetadataFetcher") as metadata_class,
                patch("sdr_harvest.pipeline.FileDownloader"),
                patch("sdr_harvest.pipeline.TextExtractor") as extractor_class,
                patch("sdr_harvest.pipeline.Chunker"),
                patch("sdr_harvest.pipeline.Embedder"),
                patch("sdr_harvest.pipeline.SolrDocumentBuilder"),
            ):
                attempts = attempts_class.return_value
                metadata = metadata_class.return_value
                metadata.fetch_cocina.return_value = (
                    cocina_path,
                    files,
                    source_fp,
                )
                extractor_class.return_value.signature.return_value = SIGNATURES[
                    "extract"
                ]
                pipeline.run_object(store.start_run("manifest.csv"), DRUID)
            self.assertEqual(
                str(Path("versions") / DRUID / source_fp),
                store.object_row(DRUID)["current_artifact_dir"],
            )
            attempts.run.assert_not_called()
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
        response.json = Mock(side_effect=AssertionError("JSON must not be parsed")) if data is None else Mock(return_value=data)
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
        self.assertEqual(
            checked_at, self.store.object_row(DRUID)["source_checked_at"]
        )
        self.assertEqual(
            0,
            self.store.db.execute(
                "SELECT count(*) FROM attempts WHERE stage='cocina'"
            ).fetchone()[0],
        )
        estimate = Pipeline(
            Settings(self.root, self.root), self.store
        )._estimate_work([DRUID], show_progress=False)
        self.assertEqual(0, estimate["cocina"])

    def test_etag_304_reuses_stored_fingerprint_and_files_without_parsing(self):
        self.seed_validators()
        response = self.response(304, etag='W/"old"', last_modified="Wed, 29 Jul 2026 19:04:31 GMT")
        self.http.get = Mock(return_value=response)
        path, files, source_fp = self.metadata.fetch_cocina(
            self.store.start_run("manifest.csv"), DRUID
        )
        self.assertEqual(self.cache, path)
        self.assertEqual(self.files, files)
        self.assertEqual(self.source_fp, source_fp)
        self.assertEqual('W/"old"', self.http.get.call_args.kwargs["headers"]["If-None-Match"])
        response.json.assert_not_called()
        self.assertIsNotNone(self.store.object_row(DRUID)["source_checked_at"])

    def test_last_modified_is_used_when_etag_is_unavailable(self):
        self.seed_validators(etag=None)
        response = self.response(304, last_modified="Wed, 29 Jul 2026 19:04:31 GMT")
        self.http.get = Mock(return_value=response)
        self.metadata.fetch_cocina(self.store.start_run("manifest.csv"), DRUID)
        headers = self.http.get.call_args.kwargs["headers"]
        self.assertEqual({"If-Modified-Since": "Wed, 29 Jul 2026 19:04:31 GMT"}, headers)
        response.json.assert_not_called()

    def test_changed_response_is_parsed_and_updates_validators(self):
        self.seed_validators()
        changed = cocina(version=2, digest="changed")
        response = self.response(200, data=changed, etag='W/"new"', last_modified="Thu, 30 Jul 2026 19:04:31 GMT")
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
                Path("/old-machine/.sdr-harvest/versions") / DRUID / "source" / "solr.json",
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
                staging_batch = SolrPublisher(
                    staging_settings, Mock()
                ).publish_batch
                first = staging.publish(
                    manifest, staging_batch, show_progress=False
                )
                second = staging.publish(
                    manifest, staging_batch, show_progress=False
                )
                store.adopt_stage(
                    DRUID,
                    "document",
                    "input",
                    "changed-output",
                    SIGNATURES["document"],
                    document,
                )
                changed = staging.publish(
                    manifest, staging_batch, show_progress=False
                )
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


if __name__ == "__main__":
    unittest.main()


class PreviousVersionReuseTest(unittest.TestCase):
    """Metadata-only changes should not redo file-derived work."""

    def write_extraction_inputs(self, version_dir: Path, content: bytes) -> None:
        version_dir.joinpath("pdfs").mkdir(parents=True)
        version_dir.joinpath("pdfs", "document.pdf").write_bytes(content)
        version_dir.joinpath("cocina.json").write_text(
            json.dumps({"type": "https://cocina.sul.stanford.edu/models/document"})
        )

    def extract(self, version_dir: Path, text: str = "PDF text") -> Path:
        with patch(
            "sdr_harvest.extract_pdf.pymupdf4llm.to_markdown", return_value=text
        ):
            return TextExtractor().run(version_dir)

    def test_download_links_verified_file_from_previous_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"pdf content"
            info = FileDownloaderTest().file_info(content)
            old = root / "old"
            old.joinpath("pdfs").mkdir(parents=True)
            old.joinpath("pdfs", "example.pdf").write_bytes(content)
            http = Mock()

            output = FileDownloader(http).run(DRUID, [info], root / "new")

            http.get.assert_not_called()
            self.assertEqual(content, output.joinpath("example.pdf").read_bytes())
            self.assertEqual(
                old.joinpath("pdfs", "example.pdf").stat().st_ino,
                output.joinpath("example.pdf").stat().st_ino,
            )

    def test_download_ignores_previous_file_with_different_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"pdf content"
            old = root / "old"
            old.joinpath("pdfs").mkdir(parents=True)
            old.joinpath("pdfs", "example.pdf").write_bytes(b"pdf CONTENT")
            http = Mock()
            http.get.return_value = StreamingResponse(content)

            output = FileDownloader(http).run(
                DRUID, [FileDownloaderTest().file_info(content)], root / "new"
            )

            http.get.assert_called_once()
            self.assertEqual(content, output.joinpath("example.pdf").read_bytes())

    def test_extraction_is_reused_when_files_and_pages_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            self.write_extraction_inputs(old, b"pdf")
            self.extract(old)
            self.write_extraction_inputs(new, b"pdf")
            extractor = TextExtractor()

            files_fp = directory_fingerprint(new / "pdfs")
            reuse = extractor.reusable_output(new, files_fp)
            with patch(
                "sdr_harvest.extract_pdf.pymupdf4llm.to_markdown"
            ) as to_markdown:
                output = extractor.run(new, files_fp, reuse_from=reuse)

            self.assertEqual(old / "markdown", reuse)
            to_markdown.assert_not_called()
            self.assertEqual("PDF text", output.joinpath("document.md").read_text())
            self.assertTrue(output.joinpath(REUSE_MARKER).exists())

    def test_extraction_is_not_reused_when_files_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            self.write_extraction_inputs(old, b"pdf")
            self.extract(old)
            self.write_extraction_inputs(new, b"changed pdf")

            self.assertIsNone(
                TextExtractor().reusable_output(
                    new, directory_fingerprint(new / "pdfs")
                )
            )

    def test_unmarked_extraction_is_reused_only_when_recorded_as_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            self.write_extraction_inputs(old, b"pdf")
            self.extract(old).joinpath(REUSE_MARKER).unlink()
            self.write_extraction_inputs(new, b"pdf")
            extractor = TextExtractor()
            files_fp = directory_fingerprint(new / "pdfs")

            self.assertIsNone(extractor.reusable_output(new, files_fp))
            self.assertEqual(
                old / "markdown",
                extractor.reusable_output(new, files_fp, old / "markdown"),
            )

    def embed(self, version_dir: Path, title: str, texts: list[str]):
        version_dir.mkdir(parents=True, exist_ok=True)
        version_dir.joinpath("metadata.json").write_text(
            json.dumps({"title_display_tesi": title})
        )
        pq.write_table(
            pa.Table.from_pylist([{"text": text} for text in texts]),
            version_dir / "chunks.parquet",
        )
        with (
            patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
            patch.object(
                Embedder,
                "embed_batch",
                side_effect=lambda batch, _key: [[0.5] * 768 for _ in batch],
            ) as embed_batch,
        ):
            Embedder().run(version_dir)
        return embed_batch

    def test_embeddings_are_reused_for_identical_retrieval_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.embed(root / "old", "Title", ["old metadata", "body"])

            embed_batch = self.embed(root / "new", "Title", ["new metadata", "body"])

            embed_batch.assert_called_once()
            self.assertEqual(
                [retrieval_document("Title", "new metadata")],
                embed_batch.call_args.args[0],
            )
            vectors = pq.read_table(root / "new" / "embeddings.parquet")
            self.assertEqual(2, vectors.num_rows)

    def test_title_change_reembeds_every_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.embed(root / "old", "Title", ["metadata", "body"])

            embed_batch = self.embed(root / "new", "New title", ["metadata", "body"])

            self.assertEqual(2, len(embed_batch.call_args.args[0]))

    def test_embeddings_with_another_signature_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.embed(root / "old", "Title", ["metadata", "body"])
            old = root / "old" / "embeddings.parquet"
            pq.write_table(
                pq.read_table(old).replace_schema_metadata(
                    {b"sdr_harvest_embed_signature": b"other-model"}
                ),
                old,
            )

            embed_batch = self.embed(root / "new", "Title", ["metadata", "body"])

            self.assertEqual(2, len(embed_batch.call_args.args[0]))

    def test_metadata_change_reuses_file_work_through_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            store.reconcile_manifest({DRUID})
            run_id = store.start_run("manifest.csv")
            content = b"pdf"
            cache = root / "sources" / DRUID / "cocina.json"
            cache.parent.mkdir(parents=True)

            def build(version: int):
                data = cocina(version, hashlib.sha1(content).hexdigest())
                data["description"]["title"][0]["value"] = f"Edition {version}"
                cache.write_text(json.dumps(data))
                files = cocina_source_files(data)
                source_fp = source_fingerprint(data, files)

                def search_metadata(_druid, version_dir):
                    output = version_dir / "metadata.json"
                    output.write_text(
                        json.dumps(
                            {"title_display_tesi": "Example", "note_tesi": str(version)}
                        )
                    )
                    return output

                pipeline = Pipeline(Settings(root, root), store)
                pipeline.http = Mock()
                pipeline.http.get.return_value = StreamingResponse(content)
                with (
                    patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                    patch.object(
                        MetadataFetcher,
                        "fetch_cocina",
                        return_value=(cache, files, source_fp),
                    ),
                    patch.object(
                        MetadataFetcher,
                        "create_search_metadata",
                        side_effect=search_metadata,
                    ),
                    patch(
                        "sdr_harvest.extract_pdf.pymupdf4llm.to_markdown",
                        return_value="PDF text",
                    ) as to_markdown,
                    patch.object(
                        Embedder,
                        "embed_batch",
                        side_effect=lambda batch, _key: [[0.5] * 768 for _ in batch],
                    ) as embed_batch,
                ):
                    pipeline.run_object(run_id, DRUID)
                return pipeline.http, to_markdown, embed_batch

            build(1)
            http, to_markdown, embed_batch = build(2)

            http.get.assert_not_called()
            to_markdown.assert_not_called()
            embed_batch.assert_called_once()
            self.assertEqual(1, len(embed_batch.call_args.args[0]))
            self.assertIn("note: 2", embed_batch.call_args.args[0][0])
            store.close()
