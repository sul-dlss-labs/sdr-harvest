import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from sdr_harvest.core import StageError
from sdr_harvest.evaluation import (
    Judgment,
    group_by_test_type,
    judged_at,
    load_judgments,
    win_tie_loss,
)
from sdr_harvest.mcp_evaluation import render_markdown, run_mcp_evaluation
from sdr_harvest.mcp_search import McpSearchClient


def _response(result):
    response = Mock()
    response.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": result}
    return response


def _tool_result(ids, *, error=False):
    return {
        "isError": error,
        "content": [{"type": "text", "text": "Catalog search failed." if error else ""}],
        "structuredContent": None if error else {
            "results": [{"id": druid, "title": f"Title {druid}"} for druid in ids]
        },
    }


INITIALIZE = {"serverInfo": {"name": "semantic-search", "version": "1.0.0"}}


class HelperTest(unittest.TestCase):
    def test_load_judgments_keeps_zero_scores_as_judged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgments.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "test_id": "one",
                            "query": "q",
                            "judgments": [
                                {"document_id": "aa123bb4567", "score": 3},
                                {"document_id": "cc123dd4567", "score": 0},
                            ],
                        }
                    ]
                )
            )

            judgment = load_judgments(path)[0]

        self.assertEqual({"aa123bb4567": 3.0}, judgment.relevant)
        self.assertEqual({"aa123bb4567", "cc123dd4567"}, judgment.judged)

    def test_judged_at_counts_any_judgment(self):
        metrics = judged_at(["a", "x", "b", "y"], frozenset({"a", "b"}), (1, 4))

        self.assertEqual({"judged@1": 1.0, "judged@4": 0.5}, metrics)

    def test_groups_by_test_type(self):
        results = [
            {"test_type": "fact_lookup", "metrics": {"ndcg@10": 1.0}},
            {"test_type": "fact_lookup", "metrics": {"ndcg@10": 0.0}},
            {"test_type": "", "metrics": {"ndcg@10": 0.4}},
        ]

        groups = group_by_test_type(results)

        self.assertEqual(2, groups["fact_lookup"]["count"])
        self.assertEqual(0.5, groups["fact_lookup"]["aggregate"]["ndcg@10"])
        self.assertEqual(1, groups["unspecified"]["count"])

    def test_win_tie_loss(self):
        a = [{"id": str(n), "metrics": {"ndcg@10": v}} for n, v in enumerate([1, 0.5, 0])]
        b = [{"id": str(n), "metrics": {"ndcg@10": v}} for n, v in enumerate([0, 0.5, 1])]

        self.assertEqual({"win": 1, "tie": 1, "loss": 1}, win_tie_loss(a, b))


class McpSearchClientTest(unittest.TestCase):
    def test_initializes_once_and_dedupes_results(self):
        http = Mock()
        http.post.side_effect = [
            _response(INITIALIZE),
            _response(_tool_result(["AA123BB4567", "aa123bb4567", "cc123dd4567"])),
            _response(_tool_result(["cc123dd4567"])),
        ]
        client = McpSearchClient("https://example.test/mcp", http)

        first = client.search("q", "keyword")
        client.search("q", "vector")

        self.assertEqual(["aa123bb4567", "cc123dd4567"], [r["druid"] for r in first])
        self.assertEqual("semantic-search", client.server_info["name"])
        self.assertEqual(3, http.post.call_count)
        arguments = http.post.call_args_list[1].kwargs["json"]["params"]["arguments"]
        self.assertEqual(
            {"query": "q", "search_type": "keyword", "rows": 20}, arguments
        )

    def test_retries_tool_errors(self):
        http = Mock()
        http.post.side_effect = [
            _response(INITIALIZE),
            _response(_tool_result([], error=True)),
            _response(_tool_result(["aa123bb4567"])),
        ]
        client = McpSearchClient("https://example.test/mcp", http, backoff=0)

        self.assertEqual("aa123bb4567", client.search("q", "hybrid")[0]["druid"])

    def test_raises_after_persistent_tool_errors(self):
        http = Mock()
        http.post.side_effect = [_response(INITIALIZE)] + [
            _response(_tool_result([], error=True)) for _ in range(2)
        ]
        client = McpSearchClient(
            "https://example.test/mcp", http, retries=1, backoff=0
        )

        with self.assertRaisesRegex(StageError, "Catalog search failed"):
            client.search("q", "vector")

    def test_rejects_unknown_search_type(self):
        client = McpSearchClient("https://example.test/mcp", Mock())

        with self.assertRaisesRegex(StageError, "Unknown search type"):
            client.search("q", "fuzzy")


class McpEvaluationTest(unittest.TestCase):
    def test_runs_each_mode_and_renders_report(self):
        rankings = {
            "keyword": ["zz000zz0000", "aa123bb4567"],
            "vector": ["aa123bb4567", "cc123dd4567"],
        }
        client = Mock()
        client.url = "https://example.test/mcp"
        client.server_info = {}
        client.search.side_effect = lambda query, mode, rows: [
            {"druid": druid, "title": druid, "score": None} for druid in rankings[mode]
        ]
        judgments = [
            Judgment(
                "case-1",
                "query text",
                {"aa123bb4567": 3.0, "cc123dd4567": 1.0},
                "known_item",
                frozenset({"aa123bb4567", "cc123dd4567"}),
            )
        ]

        report = run_mcp_evaluation(client, judgments, ("keyword", "vector"))
        markdown = render_markdown(report)

        vector = report["modes"]["vector"]
        self.assertEqual(1.0, vector["aggregate"]["ndcg@10"])
        self.assertEqual(1.0, vector["aggregate"]["judged@10"])
        self.assertEqual(0.5, report["modes"]["keyword"]["aggregate"]["judged@10"])
        self.assertIn("| known_item (1) † |", markdown)
        self.assertIn("| Vector | **1.00** |", markdown)
        self.assertIn("| Vector vs Keyword | 1 | 0 | 0 |", markdown)
        self.assertEqual(markdown, render_markdown(json.loads(json.dumps(report, sort_keys=True))))

    def test_rejects_cutoff_beyond_rows(self):
        with self.assertRaisesRegex(StageError, "exceeds"):
            run_mcp_evaluation(Mock(), [], ("vector",), cutoffs=(25,), rows=20)


if __name__ == "__main__":
    unittest.main()
