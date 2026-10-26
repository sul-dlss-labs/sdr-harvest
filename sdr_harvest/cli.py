from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
import warnings
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import requests
from urllib3.exceptions import InsecureRequestWarning

from .core import Settings, StageError
from .dark import (
    exclusive_state,
    normalize_druid,
    purge_local_state,
    read_dark_list,
    record_dark,
    remove_from_manifest,
)
from .evaluation import compare_reports, live_evaluator, load_judgments, parse_cutoffs
from .manifests import merge_manifests, parse_manifest
from .mcp_evaluation import render_markdown, run_mcp_evaluation
from .mcp_search import DEFAULT_MCP_URL, MAX_ROWS, SEARCH_TYPES, McpSearchClient
from .pipeline import Pipeline
from .publisher import CorpusPublisher, SolrPublisher
from .state import STAGES, StateStore


RESOURCE_TRACKER_WARNING_FILTER = (
    "ignore:resource_tracker:UserWarning:multiprocessing.resource_tracker"
)


def _configure_child_warning_filters() -> None:
    """Hide the harmless semaphore cleanup warning after an immediate exit."""
    existing = os.environ.get("PYTHONWARNINGS", "")
    filters = [item for item in existing.split(",") if item]
    if RESOURCE_TRACKER_WARNING_FILTER not in filters:
        filters.append(RESOURCE_TRACKER_WARNING_FILTER)
        os.environ["PYTHONWARNINGS"] = ",".join(filters)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sdr-harvest", description="Resumable SDR-to-Solr pipeline")
    result.add_argument("--state-dir", type=Path, default=Path(".sdr-harvest"))
    result.add_argument("--solr-url", default="http://localhost:8983/solr/sdr-search")
    result.add_argument(
        "--dark-list",
        type=Path,
        default=Path("dark.csv"),
        help="CSV of disavowed DRUIDs that are never built and are deleted from Solr",
    )
    commands = result.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge-manifests", help="merge and deduplicate DRUID manifests")
    merge.add_argument("inputs", type=Path, nargs="+")
    merge.add_argument("--output", type=Path, default=Path("manifest.csv"))
    publish = commands.add_parser(
        "publish", help="publish all ready Solr JSON documents to one target"
    )
    publish.add_argument("--manifest", type=Path, required=True)
    publish.add_argument(
        "--target", required=True, help="full Solr collection URL"
    )
    publish.add_argument("--workers", type=int, default=4)
    publish.add_argument(
        "--batch-size", type=int, default=25, help="maximum objects per Solr update"
    )
    publish.add_argument(
        "--max-batch-mb",
        type=float,
        default=25,
        help="maximum uncompressed Solr JSON per update",
    )
    publish.add_argument(
        "--force", action="store_true", help="republish documents already current on this target"
    )
    publish.add_argument(
        "--no-progress", action="store_true", help="disable the publication progress bar"
    )
    publish.add_argument(
        "--insecure",
        action="store_true",
        help="disable TLS certificate verification (only for a trusted local tunnel)",
    )
    evaluate = commands.add_parser(
        "evaluate", help="measure ranked retrieval against relevance judgments"
    )
    evaluate.add_argument(
        "--judgments",
        type=Path,
        required=True,
        help="test-case JSON file, or a directory of them, evaluated together",
    )
    evaluate.add_argument(
        "--test-type",
        help="restrict evaluation to test cases with this test_type (e.g. fact_lookup)",
    )
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--baseline", type=Path)
    evaluate.add_argument("--target", default="http://localhost:8983/solr/sdr-search")
    evaluate.add_argument(
        "--cutoffs", default="1,5,10", help="comma-separated ranking cutoffs"
    )
    evaluate.add_argument(
        "--candidates",
        type=int,
        default=100,
        help="number of chunk candidates to retrieve before collapsing by DRUID",
    )
    evaluate.add_argument(
        "--fail-on-regression",
        metavar="METRIC",
        help="exit nonzero when this metric drops beyond the tolerance",
    )
    evaluate.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.0,
        help="allowed absolute decrease for --fail-on-regression",
    )
    evaluate.add_argument(
        "--insecure",
        action="store_true",
        help="disable TLS certificate verification (only for a trusted local tunnel)",
    )
    evaluate_mcp = commands.add_parser(
        "evaluate-mcp",
        help="compare semantic-search-demo search types against relevance judgments",
    )
    evaluate_mcp.add_argument(
        "--judgments",
        type=Path,
        default=Path("evaluations/judgments"),
        help="test-case JSON file, or a directory of them, evaluated together",
    )
    evaluate_mcp.add_argument(
        "--test-type",
        help="restrict evaluation to test cases with this test_type (e.g. fact_lookup)",
    )
    evaluate_mcp.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    evaluate_mcp.add_argument(
        "--search-types",
        default=",".join(SEARCH_TYPES),
        help="comma-separated catalog_search_tool search types to compare",
    )
    evaluate_mcp.add_argument(
        "--cutoffs", default="1,5,10", help="comma-separated ranking cutoffs"
    )
    evaluate_mcp.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/mcp"),
        help="directory for results.json and report.md",
    )
    evaluate_mcp.add_argument(
        "--from-results",
        type=Path,
        help="re-render report.md from a saved results.json without querying",
    )
    query = commands.add_parser(
        "query", help="run one semantic query and display ranked results"
    )
    query.add_argument("query", help="natural-language query text")
    query.add_argument("--target", default="http://localhost:8983/solr/sdr-search")
    query.add_argument(
        "--limit", type=int, default=10, help="number of DRUID results to display"
    )
    query.add_argument(
        "--candidates",
        type=int,
        default=100,
        help="number of chunk candidates to retrieve before collapsing by DRUID",
    )
    query.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    query.add_argument(
        "--insecure",
        action="store_true",
        help="disable TLS certificate verification (only for a trusted local tunnel)",
    )
    for name in ("run", "plan"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name == "run":
            command.add_argument("--druid", action="append", default=[])
            command.add_argument("--workers", type=int, default=4)
            command.add_argument(
                "--keep-failed-downloads",
                action="store_true",
                help="retain downloads that fail integrity checks as .invalid files",
            )
            command.add_argument(
                "--no-progress", action="store_true", help="disable progress bars"
            )
    status = commands.add_parser("status")
    status.add_argument("--failed", action="store_true")
    status.add_argument("--druid")
    status.add_argument("--stage", choices=STAGES)
    retry = commands.add_parser("retry")
    retry.add_argument("--failed", action="store_true")
    retry.add_argument("--druid", action="append", default=[])
    retry.add_argument("--manifest", type=Path)
    retry.add_argument("--workers", type=int, default=4)
    retry.add_argument(
        "--keep-failed-downloads",
        action="store_true",
        help="retain downloads that fail integrity checks as .invalid files",
    )
    retry.add_argument(
        "--no-progress", action="store_true", help="disable progress bars"
    )
    rebuild = commands.add_parser("rebuild")
    rebuild.add_argument("--druid", required=True)
    rebuild.add_argument("--from", dest="from_stage", choices=STAGES, default="cocina")
    rebuild.add_argument("--manifest", type=Path)
    rebuild.add_argument("--workers", type=int, default=4)
    rebuild.add_argument(
        "--keep-failed-downloads",
        action="store_true",
        help="retain downloads that fail integrity checks as .invalid files",
    )
    rebuild.add_argument(
        "--no-progress", action="store_true", help="disable progress bars"
    )
    darken = commands.add_parser(
        "darken",
        help="disavow objects: record them as dark, purge local state, and delete them from Solr",
    )
    darken.add_argument(
        "--druid", action="append", default=[], help="DRUID to add to the dark list"
    )
    darken.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    darken.add_argument(
        "--target",
        dest="targets",
        metavar="URL",
        action="append",
        default=[],
        help="Solr collection URL to delete every dark object from; repeatable",
    )
    darken.add_argument(
        "--insecure",
        action="store_true",
        help="disable TLS certificate verification (only for a trusted local tunnel)",
    )
    prune = commands.add_parser("prune")
    prune.add_argument("--failed-before", required=True, help="ISO date; removes old non-current version directories")
    return result


def _manifest(args, store: StateStore) -> Path:
    value = getattr(args, "manifest", None) or store.latest_manifest()
    if not value:
        raise SystemExit("No manifest supplied and no previous run manifest is recorded")
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"Manifest does not exist: {path}")
    return path


def _settings(root: Path, args) -> Settings:
    return Settings(
        root=root,
        state_dir=args.state_dir.resolve(),
        solr_url=getattr(args, "target", None) or args.solr_url,
        verify_tls=not getattr(args, "insecure", False),
        workers=max(1, getattr(args, "workers", 4)),
        keep_failed_downloads=getattr(args, "keep_failed_downloads", False),
        publish_batch_size=max(1, getattr(args, "batch_size", 25)),
        publish_max_batch_bytes=max(
            1, int(getattr(args, "max_batch_mb", 25) * 1024 * 1024)
        ),
        dark_list=args.dark_list.resolve(),
    )


def _warn_if_insecure(settings: Settings) -> None:
    if settings.verify_tls:
        return
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)
    print(
        "Warning: TLS certificate verification is disabled for "
        f"{settings.solr_url}",
        file=sys.stderr,
        flush=True,
    )


def _darken(root: Path, args, store: StateStore) -> dict:
    """Record new dark objects, then make local state and targets agree with the list.

    Every step is idempotent, so running without --druid applies the
    committed dark list to another machine or Solr environment.
    """
    settings = _settings(root, args)
    druids = [normalize_druid(value) for value in args.druid]
    with exclusive_state(settings.state_dir):
        added = record_dark(settings.dark_list, druids, date.today().isoformat())
        dark = read_dark_list(settings.dark_list)
        summary = {
            "dark_list": str(settings.dark_list),
            "dark_objects": len(dark),
            "added": added,
            "manifest_rows_removed": 0,
            "local_state_purged": [],
            "solr_targets": [],
        }
        if not dark:
            return summary
        if args.manifest.exists():
            summary["manifest_rows_removed"] = remove_from_manifest(args.manifest, dark)
        summary["local_state_purged"] = [
            druid
            for druid in sorted(dark)
            if purge_local_state(store, settings.state_dir, druid)
        ]
    for target in args.targets:
        target_settings = replace(settings, solr_url=target)
        _warn_if_insecure(target_settings)
        with requests.Session() as http:
            SolrPublisher(target_settings, http).delete_objects(sorted(dark))
        summary["solr_targets"].append(target.rstrip("/"))
    return summary


def _pipeline(root: Path, args, store: StateStore) -> Pipeline:
    return Pipeline(_settings(root, args), store)


def _print_query_results(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return
    for rank, result in enumerate(results, 1):
        score = result.get("score")
        score_label = f" score={score:.6f}" if isinstance(score, (int, float)) else ""
        print(f"{rank}. {result['druid']}{score_label}")
        details = []
        if filename := result.get("filename"):
            details.append(str(filename))
        if (chunk_index := result.get("chunk_index")) is not None:
            details.append(f"chunk {chunk_index}")
        if details:
            print(f"   {' · '.join(details)}")
        if snippet := " ".join(str(result.get("snippet", "")).split()):
            print(
                textwrap.fill(
                    snippet,
                    width=100,
                    initial_indent="   ",
                    subsequent_indent="   ",
                )
            )


def main(argv: list[str] | None = None) -> None:
    # PyMuPDF may start multiprocessing's resource tracker during extraction.
    # It inherits this targeted filter before any worker creates a semaphore.
    _configure_child_warning_filters()
    args = parser().parse_args(argv)
    root = Path.cwd()
    args.state_dir = args.state_dir.resolve()
    store = StateStore(args.state_dir / "state.sqlite3")
    try:
        if args.command == "merge-manifests":
            try:
                summary = merge_manifests(
                    args.inputs, args.output, read_dark_list(args.dark_list)
                )
            except (FileNotFoundError, ValueError) as exc:
                raise SystemExit(str(exc)) from exc
            print(json.dumps(summary, indent=2, sort_keys=True))
            return
        if args.command == "plan":
            wanted = parse_manifest(args.manifest) - read_dark_list(args.dark_list)
            known = {r[0] for r in store.db.execute("SELECT druid FROM objects WHERE manifest_present=1")}
            failed = store.failed_druids()
            print(json.dumps({"manifest_objects": len(wanted), "new": sorted(wanted - known), "absent": sorted(known - wanted), "known_failures": sorted(wanted & failed)}, indent=2))
            return
        if args.command == "run":
            summary = _pipeline(root, args, store).run(
                args.manifest,
                only=set(args.druid) or None,
                show_progress=not args.no_progress,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"]:
                raise SystemExit(1)
            return
        if args.command == "publish":
            settings = _settings(root, args)
            _warn_if_insecure(settings)
            dark = sorted(read_dark_list(settings.dark_list))
            with requests.Session() as http:
                SolrPublisher(settings, http).delete_objects(dark)

            def publish_batch(items):
                with requests.Session() as http:
                    return SolrPublisher(settings, http).publish_batch(items)

            summary = CorpusPublisher(settings, store).publish(
                args.manifest,
                publish_batch,
                force=args.force,
                show_progress=not args.no_progress,
            )
            summary["dark_deleted"] = len(dark)
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"] or summary["not_ready"]:
                raise SystemExit(1)
            return
        if args.command == "evaluate":
            cutoffs = parse_cutoffs(args.cutoffs)
            if args.candidates < 1:
                raise StageError("Candidates must be positive")
            if args.regression_tolerance < 0:
                raise StageError("Regression tolerance cannot be negative")
            if args.fail_on_regression and not args.baseline:
                raise StageError("--fail-on-regression requires --baseline")
            verify_tls = not args.insecure
            if not verify_tls:
                warnings.filterwarnings("ignore", category=InsecureRequestWarning)
            evaluator, session = live_evaluator(args.target, verify_tls=verify_tls)
            try:
                report = evaluator.run(
                    load_judgments(args.judgments, test_type=args.test_type),
                    cutoffs=cutoffs,
                    candidate_count=args.candidates,
                )
            finally:
                session.close()
            if args.baseline:
                try:
                    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise StageError(f"Could not read baseline report: {exc}") from exc
                report["comparison"] = compare_reports(report, baseline)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
            if report.get("comparison"):
                print(
                    json.dumps(
                        {
                            name: values["delta"]
                            for name, values in report["comparison"][
                                "aggregate"
                            ].items()
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            if args.fail_on_regression:
                comparison = report.get("comparison", {}).get("aggregate", {})
                if args.fail_on_regression not in comparison:
                    raise StageError(
                        f"Unknown regression metric: {args.fail_on_regression}"
                    )
                delta = comparison[args.fail_on_regression]["delta"]
                if delta < -args.regression_tolerance:
                    print(
                        f"Regression: {args.fail_on_regression} changed by {delta:.6f}",
                        file=sys.stderr,
                    )
                    raise SystemExit(2)
            return
        if args.command == "evaluate-mcp":
            if args.from_results:
                try:
                    report = json.loads(args.from_results.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise StageError(f"Could not read results: {exc}") from exc
            else:
                search_types = tuple(
                    item.strip() for item in args.search_types.split(",") if item.strip()
                )
                unknown = sorted(set(search_types) - set(SEARCH_TYPES))
                if not search_types or unknown:
                    raise StageError(
                        f"Search types must be drawn from {', '.join(SEARCH_TYPES)}"
                    )
                judgments = load_judgments(args.judgments, test_type=args.test_type)
                with requests.Session() as http:
                    report = run_mcp_evaluation(
                        McpSearchClient(args.mcp_url, http),
                        judgments,
                        search_types,
                        cutoffs=parse_cutoffs(args.cutoffs),
                        rows=MAX_ROWS,
                    )
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / "results.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            markdown = render_markdown(report)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
            print(markdown)
            return
        if args.command == "query":
            if args.limit < 1:
                raise StageError("Limit must be positive")
            if args.candidates < args.limit:
                raise StageError(
                    f"Candidate count ({args.candidates}) must be at least the "
                    f"result limit ({args.limit})"
                )
            verify_tls = not args.insecure
            if not verify_tls:
                warnings.filterwarnings("ignore", category=InsecureRequestWarning)
            evaluator, session = live_evaluator(args.target, verify_tls=verify_tls)
            try:
                results = evaluator.query(
                    args.query, candidate_count=args.candidates
                )[: args.limit]
            finally:
                session.close()
            if args.json:
                print(
                    json.dumps(
                        {
                            "query": args.query,
                            "candidate_chunks": args.candidates,
                            "result_count": len(results),
                            "results": results,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                _print_query_results(results)
            return
        if args.command == "status":
            rows = store.rows_for_status(failed_only=args.failed, druid=args.druid, stage=args.stage)
            for row in rows:
                error = f" {row['error_category']}: {row['error_message']}" if row["error_message"] else ""
                stage = row["stage"] or "-"
                state = "absent" if not row["manifest_present"] else (row["status"] or "unprocessed")
                print(f"{row['druid']}\t{stage}\t{state}\tattempts={row['attempt_count'] or 0}{error}")
            return
        if args.command == "retry":
            selected = set(args.druid)
            if args.failed:
                selected |= store.failed_druids()
            if not selected:
                raise SystemExit("Specify --failed or at least one --druid")
            summary = _pipeline(root, args, store).run(
                _manifest(args, store),
                only=selected,
                show_progress=not args.no_progress,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"]:
                raise SystemExit(1)
            return
        if args.command == "rebuild":
            if not store.object_row(args.druid):
                raise SystemExit(f"Unknown DRUID: {args.druid}")
            store.invalidate(args.druid, args.from_stage)
            summary = _pipeline(root, args, store).run(
                _manifest(args, store),
                only={args.druid},
                show_progress=not args.no_progress,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            if summary["failed"]:
                raise SystemExit(1)
            return
        if args.command == "darken":
            print(json.dumps(_darken(root, args, store), indent=2, sort_keys=True))
            return
        if args.command == "prune":
            cutoff = datetime.fromisoformat(args.failed_before).replace(tzinfo=UTC)
            removed = 0
            versions = args.state_dir / "versions"
            for object_dir in versions.glob("*") if versions.exists() else []:
                current = store.object_row(object_dir.name)
                keep = (
                    store.resolve_path(current["current_artifact_dir"]).resolve()
                    if current and current["current_artifact_dir"]
                    else None
                )
                for version in object_dir.glob("*"):
                    if keep and version.resolve() == keep:
                        continue
                    modified = datetime.fromtimestamp(version.stat().st_mtime, UTC)
                    if modified < cutoff:
                        shutil.rmtree(version)
                        removed += 1
            print(f"Removed {removed} old non-current artifact directories.")
    except StageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        # ThreadPoolExecutor registers an interpreter-exit hook that otherwise
        # waits for active workers. State has already been marked interrupted by
        # Pipeline; close the coordinator connection and terminate immediately.
        store.close()
        print("\nInterrupted; queued work was cancelled.", file=sys.stderr, flush=True)
        os._exit(130)
    finally:
        store.close()


if __name__ == "__main__":
    main()
