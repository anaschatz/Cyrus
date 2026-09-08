"""Deterministic `main-content/1` extraction; no model-written evidence."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

from jarvis_orchestrator.research import (
    DateProvenance,
    NormalizedDocumentV1,
    normalize_research_text,
)
from jarvis_orchestrator.research_security import publication_date_from_html

MAIN_CONTENT_VERSION = "main-content/1"
_BLOCK_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre"})
_SKIP_TAGS = frozenset({"script", "style", "nav", "footer", "aside", "form", "svg", "canvas", "noscript"})


@dataclass(frozen=True, slots=True)
class MainContentV1:
    document: NormalizedDocumentV1
    block_count: int
    title: str
    publisher: str
    canonical_url: str | None
    published_at: str | None
    date_provenance: DateProvenance
    extraction_version: str = MAIN_CONTENT_VERSION

    def __post_init__(self) -> None:
        if self.extraction_version != MAIN_CONTENT_VERSION or self.block_count < 0:
            raise ValueError("main content contract is invalid")
        if len(self.title) > 512 or len(self.publisher) > 256:
            raise ValueError("main content metadata is too large")


class _MainParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.block_depth = 0
        self.current: list[str] | None = None
        self.blocks: list[str] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.metadata: dict[str, str] = {}
        self.canonical_url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value or "" for key, value in attrs}
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = True
        if tag in _BLOCK_TAGS:
            self._flush_block()
            self.block_depth += 1
            self.current = []
        if tag == "br" and self.current is not None:
            self.current.append(" ")
        if tag == "meta":
            key = (values.get("property") or values.get("name") or "").casefold()
            if key in {"og:title", "twitter:title", "author", "article:author", "og:site_name", "application-name"}:
                self.metadata.setdefault(key, values.get("content", ""))
        if tag == "link" and values.get("rel", "").casefold() == "canonical" and values.get("href"):
            self.canonical_url = values["href"]

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
        if tag in _BLOCK_TAGS and self.current is not None:
            self._flush_block()
            self.block_depth = max(0, self.block_depth - 1)
            self.current = [] if self.block_depth else None

    def _flush_block(self) -> None:
        if self.current is not None:
            normalized, _ = normalize_research_text("".join(self.current), limit_bytes=4 * 1024)
            if len(normalized) >= 20:
                self.blocks.append(normalized)
            self.current.clear()

    def close(self) -> None:
        super().close()
        self._flush_block()

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.in_title:
            self.title_parts.append(data)
        if self.current is not None:
            self.current.append(data)


def extract_main_content(body: bytes, mime_type: str) -> MainContentV1:
    try:
        value = body.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("document_encoding_invalid") from exc
    if mime_type == "text/plain":
        document = NormalizedDocumentV1.from_text(value)
        blocks = sum(1 for item in document.text.split("\n\n") if len(item) >= 20)
        return MainContentV1(document, blocks, "Untitled", "", None, None, DateProvenance.NONE)
    if mime_type not in {"text/html", "application/xhtml+xml"}:
        raise ValueError("document_mime_invalid")
    parser = _MainParser()
    parser.feed(value)
    parser.close()
    document = NormalizedDocumentV1.from_text("\n\n".join(parser.blocks))
    title_value = parser.metadata.get("og:title") or parser.metadata.get("twitter:title") or " ".join(parser.title_parts)
    title, _ = normalize_research_text(title_value, limit_bytes=512)
    publisher_value = parser.metadata.get("og:site_name") or parser.metadata.get("application-name") or parser.metadata.get("article:author") or parser.metadata.get("author") or ""
    publisher, _ = normalize_research_text(publisher_value, limit_bytes=256)
    published_at, provenance = publication_date_from_html(value)
    return MainContentV1(
        document,
        len(parser.blocks),
        title or "Untitled",
        publisher,
        parser.canonical_url,
        published_at,
        provenance,
    )
