import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from sdr_harvest.chunk import CHUNK_OVERLAP, CHUNK_SIZE
from sdr_harvest.core import (
    SIGNATURES,
    Settings,
    StageError,
)
from sdr_harvest.embed import (
    retrieval_document,
    retrieval_title,
)
from sdr_harvest.manifests import (
    cocina_caption_media_files,
    cocina_page_numbers,
    cocina_pdf_files,
    cocina_source_files,
    source_fingerprint,
)
from sdr_harvest.pipeline import Pipeline
from sdr_harvest.services import DefaultPipelineServiceFactory, PipelineServices
from sdr_harvest.stages import BUILD_STAGES, STAGE_REGISTRY, StageDefinition
from sdr_harvest.state import STAGES, StateStore
from tests.support import DRUID, cocina


class StageRegistryTest(unittest.TestCase):
    def test_registry_is_the_source_of_stage_order_and_default_signatures(self):
        self.assertEqual(STAGES, tuple(stage.name for stage in STAGE_REGISTRY))
        self.assertEqual(STAGES[1:], tuple(stage.name for stage in BUILD_STAGES))
        self.assertEqual(
            SIGNATURES,
            {stage.name: stage.default_signature for stage in STAGE_REGISTRY},
        )

    def test_dynamic_stage_uses_runtime_signature_for_currentness(self):
        stage = StageDefinition("extract", "default", dynamic_signature=True)
        record = Mock(
            status="succeeded",
            input_fingerprint="input",
            stage_signature="selected-strategy",
        )

        signature = stage.signature("selected-strategy")

        self.assertTrue(
            stage.is_current(
                record,
                "input",
                signature,
                artifact_is_current=True,
            )
        )
        self.assertFalse(
            stage.is_current(
                record,
                "input",
                signature,
                artifact_is_current=False,
            )
        )


class PipelineServicesTest(unittest.TestCase):
    def test_default_factory_wires_shared_http_state_and_closes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
            settings = Settings(root, root, keep_failed_downloads=True)
            http = Mock()
            factory = DefaultPipelineServiceFactory(
                session_factory=lambda: http,
                environment={"STACKS_TOKEN": "secret"},
            )

            with factory.create(settings, store, None) as services:
                self.assertIs(http, services.metadata.http)
                self.assertIs(http, services.downloader.http)
                self.assertIs(services.attempts, services.metadata.attempts)
                self.assertTrue(services.downloader.keep_failed_downloads)
                self.assertEqual(
                    {"Authorization": "Bearer secret"},
                    services.downloader.headers,
                )

            http.close.assert_called_once_with()
            store.close()

    def test_pipeline_closes_services_when_source_fetch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = StateStore(root / "state.sqlite3")
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
            services.metadata.fetch_cocina.side_effect = StageError("failed")
            factory = Mock()
            factory.create.return_value = services
            pipeline = Pipeline(Settings(root, root), store, service_factory=factory)

            with self.assertRaisesRegex(StageError, "failed"):
                pipeline.run_object(store.start_run("manifest.csv"), DRUID)

            closer.assert_called_once_with()
            store.close()


class FingerprintTest(unittest.TestCase):
    def test_chunk_signature_records_current_splitter_parameters(self):
        self.assertEqual(
            f"recursive-{CHUNK_SIZE}-{CHUNK_OVERLAP}-v3",
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

    def test_selects_only_world_and_stanford_files(self):
        data = cocina()
        template = data["structural"]["contains"][0]["structural"]["contains"][0]
        files = [
            {**template, "filename": f"{view}.pdf", "access": {"view": view}}
            for view in ["world", "stanford", "location-based", "dark"]
        ]
        unknown = {**template, "filename": "unknown.pdf"}
        del unknown["access"]
        data["structural"]["contains"][0]["structural"]["contains"] = files + [unknown]

        self.assertEqual(
            ["stanford.pdf", "world.pdf"],
            [file["filename"] for file in cocina_source_files(data)],
        )

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
                            "access": {"view": "world"},
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
                            "access": {"view": "world"},
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
                "access": {"view": "world"},
            }
        ]
        if include_alto:
            files.append(
                {
                    "externalIdentifier": f"file:page-alto-{number}",
                    "filename": f"page-{number}.xml",
                    "hasMimeType": "application/xml",
                    "access": {"view": "world"},
                    "use": "transcription",
                }
            )
        return {
            "externalIdentifier": f"resource:page-{number}",
            "type": "https://cocina.sul.stanford.edu/models/resources/page",
            "structural": {"contains": files},
        }

    @staticmethod
    def full_pdf_resource(
        filename: str = "full.pdf", use: str | None = None, view: str = "world"
    ) -> dict:
        pdf = {
            "externalIdentifier": f"file:{filename}",
            "filename": filename,
            "hasMimeType": "application/pdf",
            "access": {"view": view},
        }
        if use:
            pdf["use"] = use
        return {
            "type": "https://cocina.sul.stanford.edu/models/resources/object",
            "structural": {"contains": [pdf]},
        }

    @staticmethod
    def media_resource(
        number: int, captions: list[tuple[str, str, str]], kind: str = "video"
    ) -> dict:
        """A media resource with captions given as (filename, language, view)."""
        files = [
            {
                "externalIdentifier": f"file:{kind}-{number}",
                "filename": f"{kind}-{number}.mp4",
                "hasMimeType": f"{kind}/mp4",
                "access": {"view": "world"},
            }
        ]
        files += [
            {
                "externalIdentifier": f"file:{filename}",
                "filename": filename,
                "hasMimeType": "text/vtt",
                "use": "caption",
                "languageTag": language,
                "access": {"view": view},
            }
            for filename, language, view in captions
        ]
        return {
            "externalIdentifier": f"resource:{kind}-{number}",
            "type": f"https://cocina.sul.stanford.edu/models/resources/{kind}",
            "structural": {"contains": files},
        }

    def test_media_prefers_transcript_pdfs_over_captions(self):
        data = cocina()
        data["type"] = "https://cocina.sul.stanford.edu/models/media"
        data["structural"]["contains"] = [
            self.media_resource(1, [("video-1.vtt", "en", "world")]),
            self.media_resource(2, [("video-2.vtt", "en", "stanford")]),
            self.full_pdf_resource("transcript.pdf", use="transcription"),
            self.full_pdf_resource("release.pdf"),
        ]

        self.assertEqual(
            ["transcript.pdf"],
            [file["filename"] for file in cocina_source_files(data)],
        )

    def test_media_without_a_transcript_uses_its_captions(self):
        data = cocina()
        data["type"] = "https://cocina.sul.stanford.edu/models/media"
        data["structural"]["contains"] = [
            self.media_resource(1, [("video-1.vtt", "en", "world")]),
            self.media_resource(2, [("video-2.vtt", "en", "stanford")]),
            self.media_resource(3, []),
            # A PDF that is not a transcript, such as a release form.
            self.full_pdf_resource("release.pdf"),
            # A transcript the pipeline may not index.
            self.full_pdf_resource(
                "transcript.pdf", use="transcription", view="location-based"
            ),
        ]

        self.assertEqual(
            ["video-1.vtt", "video-2.vtt"],
            [file["filename"] for file in cocina_source_files(data)],
        )
        self.assertEqual(
            {"video-1.vtt": "video-1.mp4", "video-2.vtt": "video-2.mp4"},
            cocina_caption_media_files(data),
        )

    def test_captions_name_the_media_file_the_viewer_plays(self):
        data = cocina()
        resource = self.media_resource(1, [("audio.vtt", "en", "world")])
        resource["type"] = "https://cocina.sul.stanford.edu/models/resources/audio"
        resource["structural"]["contains"][0:1] = [
            {
                "filename": "master.wav",
                "hasMimeType": "audio/x-wav",
                "access": {"view": "world"},
            },
            {
                "filename": "derivative.m4a",
                "hasMimeType": "audio/mp4",
                "access": {"view": "world"},
            },
        ]
        data["structural"]["contains"] = [resource]

        self.assertEqual(
            {"audio.vtt": "derivative.m4a"}, cocina_caption_media_files(data)
        )

    def test_media_selects_one_caption_track_per_file_preferring_english(self):
        data = cocina()
        data["structural"]["contains"] = [
            self.media_resource(
                1,
                [
                    ("video-1-de.vtt", "de", "world"),
                    ("video-1-en.vtt", "en-US", "world"),
                ],
            ),
            self.media_resource(
                2, [("video-2-b.vtt", "", "world"), ("video-2-a.vtt", "", "world")]
            ),
        ]

        self.assertEqual(
            ["video-1-en.vtt", "video-2-a.vtt"],
            [file["filename"] for file in cocina_source_files(data)],
        )

    def test_ignores_captions_outside_media_resources(self):
        data = cocina()
        data["structural"]["contains"][0]["structural"]["contains"].append(
            {
                "externalIdentifier": "file:loose",
                "filename": "loose.vtt",
                "hasMimeType": "text/vtt",
                "access": {"view": "world"},
            }
        )

        self.assertEqual(
            ["example.pdf"],
            [file["filename"] for file in cocina_source_files(data)],
        )

    def test_metadata_and_file_changes_alter_source_fingerprint(self):
        first = cocina()
        changed_metadata = cocina()
        changed_metadata["description"]["title"][0]["value"] = "Changed"
        changed_file = cocina(digest="def")
        self.assertNotEqual(
            source_fingerprint(first, cocina_pdf_files(first)),
            source_fingerprint(changed_metadata, cocina_pdf_files(changed_metadata)),
        )
        self.assertNotEqual(
            source_fingerprint(first, cocina_pdf_files(first)),
            source_fingerprint(changed_file, cocina_pdf_files(changed_file)),
        )
