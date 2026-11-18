import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sdr_harvest.core import IMAGE_EXTRACT_SIGNATURE, StageError, TransientStageError
from sdr_harvest.download import DERIVATIVES_FILENAME, FileDownloader
from sdr_harvest.extract_image import (
    ImageDescriptionStrategy,
    cocina_subjects,
    cocina_title,
    describe_image,
)
from sdr_harvest.extract_text import REUSE_MARKER, TextExtractor
from sdr_harvest.manifests import (
    IIIF_LONG_EDGE,
    cocina_page_numbers,
    cocina_source_files,
)


DRUID = "tz068rz4006"
JPEG = b"\xff\xd8\xff\xe0jpeg"
IMAGE_TYPE = "https://cocina.sul.stanford.edu/models/image"


def image_file(name: str, *, width=5993, height=4806, view="world", sha1=None):
    return {
        "externalIdentifier": f"file:{name}",
        "filename": name,
        "hasMimeType": "image/jp2",
        "size": 5_000_000,
        "version": 6,
        "access": {"view": view},
        "presentation": {"width": width, "height": height},
        "hasMessageDigests": [{"type": "sha1", "digest": sha1 or f"sha1-{name}"}],
    }


def image_cocina(*files: dict, object_type: str = IMAGE_TYPE) -> dict:
    return {
        "type": object_type,
        "label": "Label",
        "description": {
            "title": [{"value": "R/V Proteus on maiden voyage, 1969"}],
            "subject": [
                {"value": "R/V Proteus (Ship)"},
                {
                    "structuredValue": [
                        {"value": "Oceanography"},
                        {"value": "California"},
                    ]
                },
            ],
        },
        "structural": {
            "contains": [
                {
                    "type": "https://cocina.sul.stanford.edu/models/resources/image",
                    "externalIdentifier": f"resource:{index}",
                    "structural": {"contains": [file]},
                }
                for index, file in enumerate(files)
            ]
        },
    }


class StreamingResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_content(self, chunk_size: int):
        yield self.content


class ImageSourceSelectionTest(unittest.TestCase):
    def test_image_object_selects_jp2s_with_iiif_size(self):
        data = image_cocina(
            image_file("b.jp2"),
            image_file("a.jp2", width=800, height=600),
            image_file("dark.jp2", view="dark"),
        )

        files = cocina_source_files(data)

        self.assertEqual(["a.jp2", "b.jp2"], [item["filename"] for item in files])
        self.assertEqual(
            ["full", f"!{IIIF_LONG_EDGE},{IIIF_LONG_EDGE}"],
            [item["iiif_size"] for item in files],
        )
        self.assertEqual("sha1-a.jp2", files[0]["sha1"])

    def test_image_resources_are_numbered_like_pages(self):
        data = image_cocina(image_file("first.jp2"), image_file("second.jp2"))

        self.assertEqual(
            {"first.jp2": "1", "second.jp2": "2"}, cocina_page_numbers(data)
        )

    def test_jp2s_of_other_object_types_are_not_selected(self):
        data = image_cocina(
            image_file("page.jp2"),
            object_type="https://cocina.sul.stanford.edu/models/book",
        )

        self.assertEqual([], cocina_source_files(data))
        self.assertEqual({}, cocina_page_numbers(data))


class DerivativeDownloadTest(unittest.TestCase):
    info = {
        "filename": "HopkinsMarineStation0000231.jp2",
        "size": 5_000_000,
        "sha1": "abc",
        "md5": "def",
        "iiif_size": "!1536,1536",
    }

    def test_downloads_iiif_jpeg_and_records_its_source(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory) / DRUID / "fp1"
            version_dir.mkdir(parents=True)
            http = Mock()
            http.get.return_value = StreamingResponse(JPEG)

            output = FileDownloader(http, stacks_token="token").run(
                DRUID, [self.info], version_dir
            )

            http.get.assert_called_once_with(
                "https://stacks.stanford.edu/image/iiif/"
                f"{DRUID}%2FHopkinsMarineStation0000231/full/!1536,1536/0/default.jpg",
                headers={"Authorization": "Bearer token"},
                timeout=120,
                stream=True,
                allow_redirects=False,
            )
            self.assertEqual(
                {"HopkinsMarineStation0000231.jpg", DERIVATIVES_FILENAME},
                {path.name for path in output.iterdir()},
            )
            self.assertEqual(
                {
                    "HopkinsMarineStation0000231.jpg": {
                        "source_file": "HopkinsMarineStation0000231.jp2",
                        "sha1": "abc",
                        "md5": "def",
                        "iiif_size": "!1536,1536",
                    }
                },
                json.loads(output.joinpath(DERIVATIVES_FILENAME).read_text()),
            )

            FileDownloader(http).run(DRUID, [self.info], version_dir)
            http.get.assert_called_once()

    def test_reuses_derivative_from_identical_earlier_version(self):
        with tempfile.TemporaryDirectory() as directory:
            http = Mock()
            http.get.return_value = StreamingResponse(JPEG)
            first = Path(directory) / DRUID / "fp1"
            first.mkdir(parents=True)
            FileDownloader(http).run(DRUID, [self.info], first)

            second = Path(directory) / DRUID / "fp2"
            second.mkdir()
            output = FileDownloader(http).run(DRUID, [self.info], second)

            http.get.assert_called_once()
            self.assertEqual(
                JPEG, output.joinpath("HopkinsMarineStation0000231.jpg").read_bytes()
            )

            third = Path(directory) / DRUID / "fp3"
            third.mkdir()
            FileDownloader(http).run(
                DRUID, [{**self.info, "iiif_size": "full"}], third
            )
            self.assertEqual(2, http.get.call_count)

    def test_rejects_a_response_that_is_not_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            http = Mock()
            http.get.return_value = StreamingResponse(b"<html>error</html>")

            with self.assertRaisesRegex(StageError, "not a JPEG"):
                FileDownloader(http).run(DRUID, [self.info], version_dir)
            self.assertEqual([], list((version_dir / "pdfs").iterdir()))


class ImageDescriptionTest(unittest.TestCase):
    def version_dir(self, root: Path, data: dict) -> Path:
        root.joinpath("cocina.json").write_text(json.dumps(data))
        source = root / "pdfs"
        source.mkdir()
        derivatives = {}
        for name in ("first", "second"):
            source.joinpath(f"{name}.jpg").write_bytes(JPEG)
            derivatives[f"{name}.jpg"] = {"source_file": f"{name}.jp2"}
        source.joinpath(DERIVATIVES_FILENAME).write_text(json.dumps(derivatives))
        return root

    def test_reads_title_and_subjects_for_prompt_context(self):
        data = image_cocina()

        self.assertEqual("R/V Proteus on maiden voyage, 1969", cocina_title(data))
        self.assertEqual(
            ["R/V Proteus (Ship)", "Oceanography -- California"],
            cocina_subjects(data),
        )

    def test_describes_each_image_as_its_own_page(self):
        data = image_cocina(image_file("first.jp2"), image_file("second.jp2"))
        with tempfile.TemporaryDirectory() as directory:
            version_dir = self.version_dir(Path(directory), data)
            extractor = TextExtractor()
            self.assertEqual(IMAGE_EXTRACT_SIGNATURE, extractor.signature(version_dir))

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "key"}),
                patch(
                    "sdr_harvest.extract_image.describe_image",
                    side_effect=lambda image, prompt, key: f"About {image.stem}",
                ) as describe,
            ):
                output = extractor.run(version_dir)

            self.assertEqual("About first", output.joinpath("first.md").read_text())
            self.assertEqual(
                {
                    "first.md": {"page": "1", "source_file": "first.jp2"},
                    "second.md": {"page": "2", "source_file": "second.jp2"},
                },
                json.loads(output.joinpath("pages.json").read_text()),
            )
            prompt = describe.call_args.args[1]
            self.assertIn("R/V Proteus on maiden voyage, 1969", prompt)
            self.assertIn("Oceanography -- California", prompt)

    def test_reuse_requires_the_same_prompt_metadata(self):
        data = image_cocina(image_file("first.jp2"), image_file("second.jp2"))
        with tempfile.TemporaryDirectory() as directory:
            versions = Path(directory)
            first = versions / "fp1"
            first.mkdir()
            self.version_dir(first, data)
            first.joinpath("markdown").mkdir()
            extractor = TextExtractor()
            provenance = extractor._provenance(first, "files")
            first.joinpath("markdown", REUSE_MARKER).write_text(
                json.dumps(provenance)
            )

            same = versions / "fp2"
            same.mkdir()
            self.version_dir(same, {**data, "label": "Changed label"})
            self.assertEqual(
                first / "markdown", extractor.reusable_output(same, "files")
            )

            retitled = versions / "fp3"
            retitled.mkdir()
            changed = json.loads(json.dumps(data))
            changed["description"]["title"] = [{"value": "New title"}]
            self.version_dir(retitled, changed)
            self.assertIsNone(extractor.reusable_output(retitled, "files"))

    def test_strategy_requires_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "first.jpg"
            image.write_bytes(JPEG)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(StageError, "LITELLM_API_KEY"):
                    ImageDescriptionStrategy().extract([image], Path(directory))


class DescribeImageTest(unittest.TestCase):
    def describe(self, response: Mock) -> str:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "first.jpg"
            image.write_bytes(JPEG)
            with patch(
                "sdr_harvest.extract_image.requests.post", return_value=response
            ) as post:
                text = describe_image(image, "Describe it", "key")
            self.sent = post.call_args.kwargs["json"]
            return text

    def test_returns_text_and_skips_thought_parts(self):
        response = Mock(status_code=200, headers={}, text="")
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking", "thought": True},
                            {"text": "A ship at anchor.\n"},
                        ]
                    }
                }
            ]
        }

        self.assertEqual("A ship at anchor.", self.describe(response))
        parts = self.sent["contents"][0]["parts"]
        self.assertEqual("image/jpeg", parts[0]["inline_data"]["mime_type"])
        self.assertEqual("Describe it", parts[1]["text"])

    def test_rate_limit_is_transient(self):
        response = Mock(
            status_code=429, headers={"Retry-After": "12"}, text="slow down"
        )

        with self.assertRaises(TransientStageError) as raised:
            self.describe(response)
        self.assertEqual(12.0, raised.exception.retry_after)

    def test_blocked_response_without_text_fails(self):
        response = Mock(status_code=200, headers={}, text="")
        response.json.return_value = {
            "candidates": [{"finishReason": "SAFETY"}]
        }

        with self.assertRaisesRegex(StageError, "finishReason=SAFETY"):
            self.describe(response)


if __name__ == "__main__":
    unittest.main()
