"""JARVIS Research sidecar adapters."""

from .adapters import (
    DirectHTTPFetcher,
    FetchResultV1,
    ResearchRetrievalPipeline,
    RetrievalCorpusV1,
    SearchCandidateV1,
    SearXNGSearchAdapter,
)
from .extraction import MainContentV1, extract_main_content

__all__ = [
    "DirectHTTPFetcher",
    "FetchResultV1",
    "ResearchRetrievalPipeline",
    "RetrievalCorpusV1",
    "SearchCandidateV1",
    "SearXNGSearchAdapter",
    "MainContentV1",
    "extract_main_content",
]
