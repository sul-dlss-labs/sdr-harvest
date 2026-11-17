from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import requests

from .attempts import StageAttempts
from .chunk import Chunker
from .core import Settings
from .create_solr_document import SolrDocumentBuilder
from .download import FileDownloader
from .embed import Embedder
from .extract_text import TextExtractor
from .metadata import MetadataFetcher
from .state import StateStore


def _noop() -> None:
    pass


@dataclass
class PipelineServices:
    """The collaborators used to process one pipeline object."""

    attempts: StageAttempts
    metadata: MetadataFetcher
    downloader: FileDownloader
    extractor: TextExtractor
    chunker: Chunker
    embedder: Embedder
    document_builder: SolrDocumentBuilder
    closer: Callable[[], None] = _noop

    def __enter__(self) -> PipelineServices:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closer()


class PipelineServiceFactory(Protocol):
    def create(
        self,
        settings: Settings,
        store: StateStore,
        pdf_executor: concurrent.futures.Executor | None,
    ) -> PipelineServices: ...


class DefaultPipelineServiceFactory:
    """Construct production services while keeping their wiring in one place."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], requests.Session] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.environment = environment

    def create(
        self,
        settings: Settings,
        store: StateStore,
        pdf_executor: concurrent.futures.Executor | None,
    ) -> PipelineServices:
        http = (self.session_factory or requests.Session)()
        environment = self.environment if self.environment is not None else os.environ
        attempts = StageAttempts(settings, store)
        return PipelineServices(
            attempts=attempts,
            metadata=MetadataFetcher(settings, store, http, attempts),
            downloader=FileDownloader(
                http,
                keep_failed_downloads=settings.keep_failed_downloads,
                stacks_token=environment.get("STACKS_TOKEN"),
            ),
            extractor=TextExtractor(pdf_executor=pdf_executor),
            chunker=Chunker(),
            embedder=Embedder(),
            document_builder=SolrDocumentBuilder(),
            closer=http.close,
        )
