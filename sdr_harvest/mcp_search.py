from __future__ import annotations

import time

import requests

from .core import StageError


DEFAULT_MCP_URL = "https://semantic-search-demo.stanford.edu/mcp"
SEARCH_TOOL = "catalog_search_tool"
SEARCH_TYPES = ("keyword", "vector", "hybrid")
MAX_ROWS = 20
PROTOCOL_VERSION = "2025-06-18"


class McpSearchClient:
    """Minimal JSON-RPC client for the semantic-search-demo MCP server."""

    def __init__(
        self,
        url: str,
        http: requests.Session,
        *,
        timeout: float = 60,
        retries: int = 3,
        backoff: float = 2.0,
    ) -> None:
        self.url = url
        self.http = http
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.server_info: dict = {}
        self._next_id = 0
        self._initialized = False

    def _call(self, method: str, params: dict) -> dict:
        self._next_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params,
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.backoff * attempt)
            try:
                response = self.http.post(
                    self.url,
                    json=payload,
                    headers={"Accept": "application/json, text/event-stream"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                continue
            if "error" in data:
                raise StageError(f"MCP {method} failed: {data['error']}")
            return data["result"]
        raise StageError(f"MCP {method} failed: {last_error}") from last_error

    def initialize(self) -> None:
        if self._initialized:
            return
        result = self._call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "sdr-harvest", "version": "1"},
            },
        )
        self.server_info = result.get("serverInfo", {})
        self._initialized = True

    def search(
        self, query: str, search_type: str, *, rows: int = MAX_ROWS
    ) -> list[dict]:
        """Return ranked parent objects for a catalog search, one per DRUID."""
        if search_type not in SEARCH_TYPES:
            raise StageError(f"Unknown search type: {search_type}")
        if not 1 <= rows <= MAX_ROWS:
            raise StageError(f"Rows must be between 1 and {MAX_ROWS}")
        self.initialize()
        params = {
            "name": SEARCH_TOOL,
            "arguments": {"query": query, "search_type": search_type, "rows": rows},
        }
        # The server intermittently reports "Catalog search failed." for
        # queries that succeed on a second attempt, so retry tool errors too.
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.backoff * attempt)
            result = self._call("tools/call", params)
            if not result.get("isError"):
                break
        else:
            text = " ".join(
                item.get("text", "") for item in result.get("content", [])
            )
            raise StageError(
                f"{SEARCH_TOOL} returned an error for {search_type} "
                f"query {query!r}: {text}"
            )
        content = result.get("structuredContent")
        if not isinstance(content, dict) or not isinstance(
            content.get("results"), list
        ):
            raise StageError(f"{SEARCH_TOOL} returned no structured results")
        return parse_results(content["results"])


def parse_results(results: list[dict]) -> list[dict]:
    ranked = []
    seen = set()
    for item in results:
        druid = str(item.get("id", "")).strip().lower()
        if not druid or druid in seen:
            continue
        seen.add(druid)
        ranked.append(
            {"druid": druid, "title": item.get("title"), "score": item.get("score")}
        )
    return ranked
