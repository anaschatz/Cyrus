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

__version__ = "1.0.0"

__all__ = [
    "DirectHTTPFetcher",
    "FetchResultV1",
    "MainContentV1",
    "ResearchRetrievalPipeline",
    "RetrievalCorpusV1",
    "SearXNGSearchAdapter",
    "SearchCandidateV1",
    "extract_main_content",
]
