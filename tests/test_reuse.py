import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from sdr_harvest.core import (
    Settings,
    directory_fingerprint,
)
from sdr_harvest.download import FileDownloader
from sdr_harvest.embed import (
    Embedder,
    retrieval_document,
)
from sdr_harvest.extract_text import REUSE_MARKER, TextExtractor
from sdr_harvest.manifests import (
    cocina_source_files,
    source_fingerprint,
)
from sdr_harvest.metadata import MetadataFetcher
from sdr_harvest.pipeline import Pipeline
from sdr_harvest.services import DefaultPipelineServiceFactory
from sdr_harvest.state import StateStore
from tests.support import DRUID, StreamingResponse, cocina, file_info


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
            info = file_info(content)
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

            output = FileDownloader(http).run(DRUID, [file_info(content)], root / "new")

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

                http = Mock()
                http.get.return_value = StreamingResponse(content)
                pipeline = Pipeline(
                    Settings(root, root),
                    store,
                    service_factory=DefaultPipelineServiceFactory(
                        session_factory=lambda: http
                    ),
                )
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
                return http, to_markdown, embed_batch

            build(1)
            http, to_markdown, embed_batch = build(2)

            http.get.assert_not_called()
            to_markdown.assert_not_called()
            embed_batch.assert_called_once()
            self.assertEqual(1, len(embed_batch.call_args.args[0]))
            self.assertIn("note: 2", embed_batch.call_args.args[0][0])
            store.close()
