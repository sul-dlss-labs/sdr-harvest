import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf
import pymupdf4llm

from sdr_harvest.chunk import CHUNK_OVERLAP, CHUNK_SIZE, Chunker
from sdr_harvest.core import (
    ALTO_EXTRACT_SIGNATURE,
    PDF_EXTRACT_SIGNATURE,
    StageError,
)
from sdr_harvest.create_solr_document import SolrDocumentBuilder
from sdr_harvest.download import FileDownloader
from sdr_harvest.extract_pdf import PdfExtractionStrategy, extract_pdf_to_markdown
from sdr_harvest.extract_text import TextExtractor
from tests.support import (
    DRUID,
    StreamingResponse,
    cocina,
    paged_cocina,
    write_page_files,
)


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
                json.dumps({"type": "https://cocina.sul.stanford.edu/models/document"})
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
                                                "access": {"view": "world"},
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
                json.dumps({"type": "https://cocina.sul.stanford.edu/models/document"})
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
                json.dumps({"type": "https://cocina.sul.stanford.edu/models/document"})
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
            text_rows = [row for row in rows if row["pages"] is not None]

            self.assertTrue(all(row["file"] == "document.md" for row in text_rows))
            self.assertEqual({"1", "2"}, {row["pages"][0] for row in text_rows})
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

    def test_skips_pages_pymupdf4llm_cannot_convert(self):
        """Regression test for sw321tg6624 failing on one oversized page."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "document.pdf"
            document = pymupdf.open()
            for _ in range(3):
                document.new_page()
            document.save(pdf)
            document.close()

            def to_markdown(path, pages=None, **options):
                if pages is None or pages == [1]:
                    raise IndexError("index -1 is out of bounds")
                return [
                    {
                        "text": f"page {pages[0] + 1} ",
                        "metadata": {"page_number": pages[0] + 1},
                    }
                ]

            with patch(
                "sdr_harvest.extract_pdf.pymupdf4llm.to_markdown",
                side_effect=to_markdown,
            ):
                extracted = extract_pdf_to_markdown(pdf, root / "document.md")

            self.assertEqual("page 1 page 3 ", (root / "document.md").read_text())
            self.assertEqual(
                [
                    {"page": "1", "start": 0, "end": 7},
                    {"page": "2", "start": 7, "end": 7},
                    {"page": "3", "start": 7, "end": 14},
                ],
                extracted["document.md"]["pages"],
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
                headers={},
                timeout=120,
                stream=True,
                allow_redirects=False,
            )

    def test_redirect_reports_authentication_required(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            location = f"https://stacks.stanford.edu/file/auth/{DRUID}/example.pdf"
            http = Mock()
            http.get.return_value = StreamingResponse(
                b"", status_code=302, headers={"Location": location}
            )

            with self.assertRaisesRegex(
                StageError, "Stacks requires authentication for example.pdf"
            ) as raised:
                FileDownloader(http).run(DRUID, [self.file_info(b"pdf")], version_dir)

            self.assertIn(f"HTTP 302 redirect to {location}", str(raised.exception))
            self.assertFalse(version_dir.joinpath("pdfs", "example.pdf").exists())
            self.assertFalse(version_dir.joinpath("pdfs", "example.pdf.tmp").exists())

    def test_sends_stacks_token(self):
        with tempfile.TemporaryDirectory() as directory:
            content = b"pdf content"
            http = Mock()
            http.get.return_value = StreamingResponse(content)

            FileDownloader(http, stacks_token="secret").run(
                DRUID, [self.file_info(content)], Path(directory)
            )

            self.assertEqual(
                {"Authorization": "Bearer secret"},
                http.get.call_args.kwargs["headers"],
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
                FileDownloader(http).run(DRUID, [self.file_info(expected)], version_dir)

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
            self.assertFalse(version_dir.joinpath("pdfs", "example.pdf.tmp").exists())
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
            self.assertTrue(all(len(row["text"]) <= CHUNK_SIZE for row in text_rows))
            self.assertTrue(all(len(row["text"]) > CHUNK_OVERLAP for row in text_rows))
            self.assertGreater(max(len(row["text"]) for row in text_rows), 1_000)
            for first, second in zip(text_rows, text_rows[1:], strict=False):
                self.assertTrue(
                    set(first["text"].split()[-30:]) & set(second["text"].split()[:30])
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

            self.assertEqual(["42"], row["pages"])
            self.assertIsNone(row["page_start_offsets"])
            self.assertEqual("page.xml", row["source_file"])

    def test_records_where_each_page_starts_in_a_chunk_spanning_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("metadata.json").write_text("{}")
            markdown = version_dir / "markdown"
            markdown.mkdir()
            first = "alpha " * 100 + "91 "
            second = "\n" + "beta " * 100
            markdown.joinpath("document.md").write_text(first + second)
            markdown.joinpath("pages.json").write_text(
                json.dumps(
                    {
                        "document.md": {
                            "pages": [
                                {"page": "91", "start": 0, "end": len(first)},
                                {"page": "", "start": len(first), "end": len(first)},
                                {
                                    "page": "92",
                                    "start": len(first),
                                    "end": len(first) + len(second),
                                },
                            ],
                            "source_file": "document.pdf",
                        }
                    }
                )
            )

            rows = pq.read_table(Chunker().run(DRUID, version_dir)).to_pylist()
            row = next(row for row in rows if row["file"] == "document.md")

            self.assertEqual(["91", "92"], row["pages"])
            self.assertEqual([0, len(first)], row["page_start_offsets"])
            self.assertTrue(
                row["text"][row["page_start_offsets"][1] :].startswith("\nbeta")
            )

    def test_overlaps_chunks_across_consecutive_page_files(self):
        first = "".join(f"alpha line {index}\n" for index in range(150))
        first += "An additional $22"
        second = "million includes other services\n"
        second += "".join(f"beta line {index}\n" for index in range(150))
        for suffix in ("xml", "pdf"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                version_dir = Path(directory)
                names = [f"page-1.{suffix}", f"page-2.{suffix}"]
                write_page_files(
                    version_dir,
                    paged_cocina(*((name, "world") for name in names)),
                    {names[0]: ("1", first), names[1]: ("2", second)},
                )

                rows = pq.read_table(Chunker().run(DRUID, version_dir)).to_pylist()
                text_rows = [row for row in rows if row["file"] != "_metadata_"]
                seam = next(row for row in text_rows if "$22\nmillion" in row["text"])

                self.assertEqual("page-1.md", seam["file"])
                self.assertEqual(names[0], seam["source_file"])
                self.assertEqual(["1", "2"], seam["pages"])
                self.assertTrue(
                    seam["text"][seam["page_start_offsets"][1] :].startswith(
                        "million"
                    )
                )
                # Chunks are numbered within the file they start in, so child
                # document IDs stay unique.
                for name in ("page-1.md", "page-2.md"):
                    self.assertEqual(
                        list(range(sum(row["file"] == name for row in text_rows))),
                        [row["chunk_index"] for row in text_rows if row["file"] == name],
                    )

    def test_ends_a_page_sequence_where_access_changes_or_a_page_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            # Page 3 has no indexable file, so nothing was extracted for it.
            write_page_files(
                version_dir,
                paged_cocina(
                    ("page-1.xml", "world"),
                    ("page-2.xml", "stanford"),
                    ("page-3.xml", "dark"),
                    ("page-4.xml", "stanford"),
                ),
                {
                    "page-1.xml": ("1", "public " * 50),
                    "page-2.xml": ("2", "restricted " * 50),
                    "page-4.xml": ("4", "later " * 50),
                },
            )

            rows = pq.read_table(Chunker().run(DRUID, version_dir)).to_pylist()
            text_rows = [row for row in rows if row["file"] != "_metadata_"]

            self.assertEqual(
                [("page-1.md", ["1"]), ("page-2.md", ["2"]), ("page-4.md", ["4"])],
                [(row["file"], row["pages"]) for row in text_rows],
            )
            self.assertTrue(
                all(len(set(row["text"].split())) == 1 for row in text_rows)
            )

    def test_leaves_page_unknown_without_page_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("metadata.json").write_text("{}")
            markdown = version_dir / "markdown"
            markdown.mkdir()
            markdown.joinpath("page-10.md").write_text("Text without a known page")

            rows = pq.read_table(Chunker().run(DRUID, version_dir)).to_pylist()
            row = next(row for row in rows if row["file"] == "page-10.md")

            self.assertIsNone(row["pages"])

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

            self.assertIsNone(row["pages"])


class SolrDocumentBuilderTest(unittest.TestCase):
    def test_records_chunk_pages_on_child_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("metadata.json").write_text("{}")
            version_dir.joinpath("cocina.json").write_text("{}")
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "file": "_metadata_",
                            "source_file": "_metadata_",
                            "pages": None,
                            "page_start_offsets": None,
                            "chunk_index": 0,
                            "text": "Metadata",
                            "embedding": [0.1],
                        },
                        {
                            "file": "document.md",
                            "source_file": "document.pdf",
                            "pages": ["1"],
                            "page_start_offsets": None,
                            "chunk_index": 0,
                            "text": "First page",
                            "embedding": [0.2],
                        },
                        {
                            "file": "document.md",
                            "source_file": "document.pdf",
                            "pages": ["91", "92"],
                            "page_start_offsets": [0, 4],
                            "chunk_index": 1,
                            "text": "91 \nSecond page",
                            "embedding": [0.3],
                        },
                    ]
                ),
                version_dir / "embeddings.parquet",
            )

            output = SolrDocumentBuilder().run(DRUID, "source", version_dir)
            children = json.loads(output.read_text())["_childDocuments_"]

            self.assertNotIn("page_ss", children[0])
            self.assertEqual(["1", "91"], [child["page_ss"] for child in children[1:]])
            self.assertNotIn("pages_ism", children[0])
            self.assertEqual([1], children[1]["pages_ism"])
            self.assertEqual([0], children[1]["page_start_offsets_ism"])
            self.assertNotIn("page_start_offsets_ism", children[0])
            self.assertEqual([91, 92], children[2]["pages_ism"])
            self.assertEqual([0, 4], children[2]["page_start_offsets_ism"])
            self.assertEqual(3, len({child["id"] for child in children}))

    def test_sets_public_bsi_from_file_access_view(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            version_dir.joinpath("metadata.json").write_text("{}")
            data = cocina()
            data["structural"]["contains"][0]["structural"]["contains"] = [
                {"filename": "open.pdf", "access": {"view": "world"}},
                {"filename": "sunet.pdf", "access": {"view": "stanford"}},
            ]
            version_dir.joinpath("cocina.json").write_text(json.dumps(data))
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "file": name,
                            "source_file": source,
                            "pages": None,
                            "chunk_index": 0,
                            "text": "Text",
                            "embedding": [0.1],
                        }
                        for name, source in [
                            ("_metadata_", "_metadata_"),
                            ("open.md", "open.pdf"),
                            ("sunet.md", "sunet.pdf"),
                        ]
                    ]
                ),
                version_dir / "embeddings.parquet",
            )

            output = SolrDocumentBuilder().run(DRUID, "source", version_dir)
            children = json.loads(output.read_text())["_childDocuments_"]

            self.assertEqual(
                {"_metadata_": True, "open.pdf": True, "sunet.pdf": False},
                {child["filename_ss"]: child["public_bsi"] for child in children},
            )
