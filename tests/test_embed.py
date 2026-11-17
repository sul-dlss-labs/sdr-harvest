import json
import multiprocessing
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from sdr_harvest.core import (
    StageError,
    TransientStageError,
)
from sdr_harvest.embed import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CONCURRENCY,
    ERROR_BODY_LIMIT,
    RATE_LIMIT_MAX_WAIT,
    RATE_LIMIT_RESET_BUFFER,
    Embedder,
    RateLimitGate,
    create_shared_limits,
    http_error_message,
    retry_after_seconds,
)
from tests.support import DRUID


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
            retry_after_seconds(response, now=datetime(2026, 8, 18, 17, 0, tzinfo=UTC)),
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
                self.assertEqual("models/gemini-embedding-2", item["model"])
                self.assertEqual(768, item["outputDimensionality"])
            self.assertEqual(
                raw_texts,
                pq.read_table(output).column("text").to_pylist(),
            )
            vector_starts = [
                vector[0]
                for vector in pq.read_table(output).column("embedding").to_pylist()
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

    def test_waits_for_rate_limit_reset_instead_of_failing(self):
        with tempfile.TemporaryDirectory() as directory:
            version_dir = Path(directory)
            (version_dir / "metadata.json").write_text("{}")
            pq.write_table(
                pa.Table.from_pylist([{"text": "One chunk"}]),
                version_dir / "chunks.parquet",
            )
            limited = Mock(
                status_code=429, headers={"Retry-After": "30"}, text="limited"
            )
            succeeded = Mock(
                status_code=200,
                **{"json.return_value": {"embeddings": [{"values": [0.1] * 768}]}},
            )
            clock = [1_000.0]

            def sleep(seconds):
                clock[0] += seconds

            with (
                patch.dict(os.environ, {"LITELLM_API_KEY": "test-key"}),
                patch(
                    "sdr_harvest.embed.requests.post",
                    side_effect=[limited, limited, succeeded],
                ) as post,
                patch("sdr_harvest.embed._rate_limit", RateLimitGate()),
                patch("sdr_harvest.embed.time.monotonic", lambda: clock[0]),
                patch("sdr_harvest.embed.time.sleep", side_effect=sleep) as slept,
            ):
                output = Embedder().run(version_dir)

            self.assertEqual(3, post.call_count)
            self.assertEqual(
                [30.0 + RATE_LIMIT_RESET_BUFFER] * 2,
                [call.args[0] for call in slept.call_args_list],
            )
            self.assertEqual(1, pq.read_table(output).num_rows)

    def test_rate_limit_wait_is_bounded(self):
        limited = Mock(status_code=429, headers={"Retry-After": "60"}, text="limited")
        clock = [1_000.0]

        def sleep(seconds):
            clock[0] += seconds

        with (
            patch("sdr_harvest.embed.requests.post", return_value=limited),
            patch("sdr_harvest.embed._rate_limit", RateLimitGate()),
            patch("sdr_harvest.embed.time.monotonic", lambda: clock[0]),
            patch("sdr_harvest.embed.time.sleep", side_effect=sleep),
            self.assertRaisesRegex(TransientStageError, "LiteLLM HTTP 429") as raised,
        ):
            Embedder().embed_batch(["text"], "test-key")

        self.assertEqual(60.0, raised.exception.retry_after)
        self.assertGreaterEqual(clock[0] - 1_000.0, RATE_LIMIT_MAX_WAIT)

    def test_rate_limit_pauses_requests_from_other_threads(self):
        gate = RateLimitGate()
        clock = [1_000.0]
        with patch("sdr_harvest.embed.time.monotonic", lambda: clock[0]):
            gate.close_for(30.0)
            with patch(
                "sdr_harvest.embed.time.sleep",
                side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            ) as slept:
                gate.wait()
                gate.wait()

        slept.assert_called_once_with(30.0)

    def test_shared_rate_limit_is_seen_by_every_gate_using_it(self):
        _slots, open_at = create_shared_limits(multiprocessing.get_context("spawn"))
        first, second = RateLimitGate(open_at), RateLimitGate(open_at)
        clock = [1_000.0]
        with patch("sdr_harvest.embed.time.monotonic", lambda: clock[0]):
            first.close_for(30.0)
            with patch(
                "sdr_harvest.embed.time.sleep",
                side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            ) as slept:
                second.wait()

        slept.assert_called_once_with(30.0)

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
                    pa.Table.from_pylist([{"text": "first"}, {"text": second_text}]),
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
