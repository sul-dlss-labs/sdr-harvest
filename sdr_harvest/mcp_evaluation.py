from __future__ import annotations

from datetime import UTC, datetime
from itertools import combinations

from .core import StageError
from .evaluation import (
    DEFAULT_CUTOFFS,
    Judgment,
    aggregate_metrics,
    group_by_test_type,
    judged_at,
    query_metrics,
    win_tie_loss,
)
from .mcp_search import MAX_ROWS, SEARCH_TOOL, McpSearchClient


REPORT_VERSION = 1
PRIMARY_METRIC = "ndcg@10"
METRIC_LABEL = "nDCG@10"
BEST_MARGIN = 0.02
ANECDOTAL_MAX = 2
NOTABLE_QUERIES = 5
MODE_LABELS = {"keyword": "Keyword", "vector": "Vector", "hybrid": "Hybrid"}
OVERALL_METRICS = (
    ("ndcg@5", "nDCG@5"),
    ("ndcg@10", "nDCG@10"),
    ("mrr@10", "MRR@10"),
    ("recall@10", "Recall@10"),
    ("success@1", "Success@1"),
)


def run_mcp_evaluation(
    client: McpSearchClient,
    judgments: list[Judgment],
    search_types: tuple[str, ...],
    *,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    rows: int = MAX_ROWS,
) -> dict:
    if max(cutoffs) > rows:
        raise StageError(
            f"Largest cutoff ({max(cutoffs)}) exceeds the result rows ({rows})"
        )
    modes = {}
    for search_type in search_types:
        queries = []
        for judgment in judgments:
            ranked = client.search(judgment.query, search_type, rows=rows)
            druids = [item["druid"] for item in ranked]
            metrics = query_metrics(druids, judgment.relevant, cutoffs)
            metrics.update(judged_at(druids, judgment.judged, cutoffs))
            queries.append(
                {
                    "id": judgment.id,
                    "query": judgment.query,
                    "test_type": judgment.test_type,
                    "relevant": judgment.relevant,
                    "judged": sorted(judgment.judged),
                    "metrics": metrics,
                    "ranked": druids,
                    "results": ranked[: max(cutoffs)],
                }
            )
        modes[search_type] = {
            "aggregate": aggregate_metrics(queries),
            "by_test_type": group_by_test_type(queries),
            "queries": queries,
        }
    return {
        "report_version": REPORT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "retrieval": {
            "mcp_url": client.url,
            "server": client.server_info,
            "tool": SEARCH_TOOL,
            "search_types": list(search_types),
            "rows": rows,
            "cutoffs": list(cutoffs),
        },
        "query_count": len(judgments),
        "modes": modes,
    }


def _label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _best_cells(values: dict[str, float]) -> tuple[dict[str, str], str]:
    """Format one value per mode, bolding the leader, and name the winner."""
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
    top_mode, top = ordered[0]
    clear_winner = len(ordered) == 1 or top - ordered[1][1] >= BEST_MARGIN
    cells = {
        mode: f"**{_fmt(value)}**" if top - value < BEST_MARGIN else _fmt(value)
        for mode, value in values.items()
    }
    return cells, _label(top_mode) if clear_winner else "tie"


def _first_top_grade_rank(query: dict) -> int | None:
    relevant = query["relevant"]
    top = max(relevant.values())
    return next(
        (
            rank
            for rank, druid in enumerate(query["ranked"], 1)
            if relevant.get(druid) == top
        ),
        None,
    )


def render_markdown(report: dict) -> str:
    retrieval = report["retrieval"]
    # results.json is written with sorted keys, so keep the requested order.
    modes = retrieval.get("search_types") or list(report["modes"])
    rows = retrieval["rows"]
    count = report["query_count"]
    lines = [
        "# Search mode comparison: semantic-search-demo",
        "",
        f"- Run: {report['created_at'][:10]} against `{retrieval['mcp_url']}` "
        f"(`{retrieval['tool']}`, {rows} rows, no filters)",
        f"- Queries: {count}, scored at cutoffs "
        f"{', '.join(str(c) for c in retrieval['cutoffs'])}",
        "- Relevance: graded 0–3. nDCG uses gain 2^g−1; MRR, recall and success "
        "count any grade ≥ 1 as relevant. Unjudged results score 0.",
        "",
        "## Overall",
        "",
    ]

    overall_rows = []
    columns = {
        name: _best_cells(
            {mode: report["modes"][mode]["aggregate"][name] for mode in modes}
        )[0]
        for name, _ in OVERALL_METRICS
    }
    for mode in modes:
        aggregate = report["modes"][mode]["aggregate"]
        overall_rows.append(
            [_label(mode)]
            + [columns[name][mode] for name, _ in OVERALL_METRICS]
            + [_fmt(aggregate.get("judged@10", 0.0))]
        )
    lines += _table(
        ["Mode"] + [label for _, label in OVERALL_METRICS] + ["Judged@10"],
        overall_rows,
    )

    lines += [
        "",
        f"## Best mode by test type ({METRIC_LABEL})",
        "",
    ]
    test_types = sorted(
        report["modes"][modes[0]]["by_test_type"].items(),
        key=lambda item: (-item[1]["count"], item[0]),
    )
    type_rows = []
    for test_type, group in test_types + [("All", {"count": count})]:
        if test_type == "All":
            values = {
                mode: report["modes"][mode]["aggregate"][PRIMARY_METRIC]
                for mode in modes
            }
        else:
            values = {
                mode: report["modes"][mode]["by_test_type"][test_type]["aggregate"][
                    PRIMARY_METRIC
                ]
                for mode in modes
            }
        cells, best = _best_cells(values)
        marker = " †" if test_type != "All" and group["count"] <= ANECDOTAL_MAX else ""
        name = f"**All** ({count})" if test_type == "All" else (
            f"{test_type} ({group['count']}){marker}"
        )
        type_rows.append([name] + [cells[mode] for mode in modes] + [best])
    lines += _table(
        ["test_type (n)"] + [_label(mode) for mode in modes] + ["Best"], type_rows
    )
    lines += [
        "",
        f"Bold marks the leader (within {BEST_MARGIN}); “tie” means no mode leads "
        f"by ≥ {BEST_MARGIN}. † n ≤ {ANECDOTAL_MAX}, anecdotal.",
    ]

    if len(modes) > 1:
        lines += ["", f"## Head-to-head ({METRIC_LABEL}, per query)", ""]
        pair_rows = []
        for first, second in combinations(modes, 2):
            a, b = (second, first)
            counts = win_tie_loss(
                report["modes"][a]["queries"],
                report["modes"][b]["queries"],
                PRIMARY_METRIC,
            )
            pair_rows.append(
                [
                    f"{_label(a)} vs {_label(b)}",
                    str(counts["win"]),
                    str(counts["tie"]),
                    str(counts["loss"]),
                ]
            )
        lines += _table(["Pair", "Wins", "Ties", "Losses"], pair_rows)

        by_id = {
            mode: {query["id"]: query for query in report["modes"][mode]["queries"]}
            for mode in modes
        }
        spreads = []
        for query in report["modes"][modes[0]]["queries"]:
            scores = [by_id[mode][query["id"]]["metrics"][PRIMARY_METRIC] for mode in modes]
            spread = max(scores) - min(scores)
            if spread > 0:
                spreads.append((spread, query["id"]))
        spreads.sort(key=lambda item: (-item[0], item[1]))
        if spreads:
            lines += [
                "",
                "## Largest disagreements",
                "",
                f"{METRIC_LABEL} per mode, with the rank "
                f"of the first top-graded document in parentheses (– = not in top {rows}).",
                "",
            ]
            notable_rows = []
            for _, query_id in spreads[:NOTABLE_QUERIES]:
                query = by_id[modes[0]][query_id]
                text = query["query"]
                if len(text) > 70:
                    text = text[:67].rstrip() + "…"
                cells = []
                for mode in modes:
                    entry = by_id[mode][query_id]
                    rank = _first_top_grade_rank(entry)
                    cells.append(
                        f"{_fmt(entry['metrics'][PRIMARY_METRIC])} "
                        f"({rank if rank else '–'})"
                    )
                notable_rows.append(
                    [f"`{query_id}`", query["test_type"] or "–", text.replace("|", "\\|")]
                    + cells
                )
            lines += _table(
                ["Test", "Type", "Query"] + [_label(mode) for mode in modes],
                notable_rows,
            )

    judged = ", ".join(
        f"{_label(mode)} {_fmt(report['modes'][mode]['aggregate'].get('judged@10', 0.0))}"
        for mode in modes
    )
    small = sum(
        1
        for _, group in test_types
        if group["count"] <= ANECDOTAL_MAX
    )
    lines += [
        "",
        "## Caveats",
        "",
        f"- Small samples: {small} of {len(test_types)} test types have "
        f"≤ {ANECDOTAL_MAX} queries; treat their rows as anecdotes, not trends.",
        f"- Judgment coverage (Judged@10): {judged}. A mode whose hits were not "
        "in the judgment pool is penalised for unjudged results, not necessarily "
        "for bad ones.",
        f"- The MCP tool returns at most {MAX_ROWS} objects, so recall is bounded "
        "for queries with many relevant documents.",
        "",
    ]
    return "\n".join(lines)
