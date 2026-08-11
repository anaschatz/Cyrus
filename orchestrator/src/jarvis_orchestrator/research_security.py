"""Pure qualification rules for Research v1 network and metadata boundaries."""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urlsplit

from .research import DateProvenance, validate_public_url

FIXED_USER_AGENT = "JARVISResearch/1"
ROBOTS_DEADLINE_SECONDS = 3.0
ROBOTS_MAX_BYTES = 128 * 1024
ROBOTS_MAX_REDIRECTS = 2
DIRECT_MAX_REDIRECTS = 3
DIRECT_MAX_BODY_BYTES = 2 * 1024 * 1024
TRANSIENT_STATUS = frozenset({429, 502, 503, 504})
ALLOWED_MIME = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
ALLOWED_BROWSER_RESOURCE_TYPES = frozenset({"document", "stylesheet", "script", "xhr", "fetch"})
_SECRET = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b|\b(?:sk|pk|api)[-_][a-zA-Z0-9_-]{20,}\b)"
)


def validate_resolved_addresses(addresses: Iterable[str]) -> tuple[str, ...]:
    """Reject the complete DNS answer when any member is not globally routable."""

    result: list[str] = []
    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError as exc:
            raise ValueError("dns_answer_invalid") from exc
        if not address.is_global:
            raise ValueError("dns_answer_forbidden")
        result.append(address.compressed)
    if not result:
        raise ValueError("dns_answer_empty")
    return tuple(sorted(set(result), key=lambda item: (ipaddress.ip_address(item).version, item)))


def validate_egress_target(url: str, resolved_addresses: Iterable[str]) -> tuple[str, tuple[str, ...]]:
    safe = validate_public_url(url)
    parsed = urlsplit(safe)
    if parsed.port not in {None, 80, 443}:
        raise ValueError("egress_port_forbidden")
    return safe, validate_resolved_addresses(resolved_addresses)


def reject_sensitive_text(value: str) -> None:
    if _SECRET.search(value):
        raise ValueError("sensitive_value_rejected")


@dataclass(slots=True)
class RetryBudgetV1:
    used: bool = False

    def consume(
        self,
        *,
        error_code: str = "",
        status: int | None = None,
        body_bytes_received: int = 0,
        delay_seconds: float = 0.0,
        remaining_seconds: float = 0.0,
    ) -> bool:
        transient = error_code in {"connect_timeout", "connect_reset"} or status in TRANSIENT_STATUS
        if (
            self.used
            or not transient
            or body_bytes_received != 0
            or not 0 <= delay_seconds <= 0.5
            or delay_seconds >= remaining_seconds
        ):
            return False
        self.used = True
        return True


class RobotsDecision(str):
    ALLOW = "ALLOW"
    DISALLOW = "DISALLOW"
    FAIL_CLOSED = "FAIL_CLOSED"


def robots_decision(*, status: int | None, body: bytes, path: str, user_agent: str = FIXED_USER_AGENT) -> str:
    if status in {404, 410}:
        return RobotsDecision.ALLOW
    if status is None or status < 200 or status >= 300 or len(body) > ROBOTS_MAX_BYTES:
        return RobotsDecision.FAIL_CLOSED
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return RobotsDecision.FAIL_CLOSED
    groups: list[tuple[list[str], list[tuple[str, str]]]] = []
    agents: list[str] = []
    rules: list[tuple[str, str]] = []
    saw_rule = False
    for raw_line in text.splitlines():
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        if ":" not in line:
            return RobotsDecision.FAIL_CLOSED
        key, value = (item.strip() for item in line.split(":", 1))
        key = key.casefold()
        if key == "user-agent":
            if saw_rule:
                groups.append((agents, rules))
                agents, rules, saw_rule = [], [], False
            if not value:
                return RobotsDecision.FAIL_CLOSED
            agents.append(value.casefold())
        elif key in {"allow", "disallow"}:
            if not agents:
                return RobotsDecision.FAIL_CLOSED
            rules.append((key, value))
            saw_rule = True
        elif key in {"sitemap", "crawl-delay", "host", "request-rate", "visit-time"}:
            continue
        else:
            return RobotsDecision.FAIL_CLOSED
    if agents:
        groups.append((agents, rules))
    agent = user_agent.casefold()
    exact = [rules for agents, rules in groups if any(item != "*" and item in agent for item in agents)]
    selected = exact if exact else [rules for agents, rules in groups if "*" in agents]
    matching: list[tuple[int, str]] = []
    for group in selected:
        for kind, pattern in group:
            if not pattern and kind == "disallow":
                continue
            anchored = pattern.endswith("$")
            normalized = pattern[:-1] if anchored else pattern
            expression = "^" + ".*".join(re.escape(part) for part in normalized.split("*"))
            if anchored:
                expression += "$"
            if re.match(expression, path):
                matching.append((len(normalized.replace("*", "")), kind))
    if not matching:
        return RobotsDecision.ALLOW
    longest = max(length for length, _ in matching)
    kinds = {kind for length, kind in matching if length == longest}
    return RobotsDecision.ALLOW if "allow" in kinds else RobotsDecision.DISALLOW


@dataclass(frozen=True, slots=True)
class BrowserLimitsV1:
    page_max_requests: int = 40
    page_max_bytes: int = 8 * 1024 * 1024
    page_max_seconds: int = 15
    run_max_requests: int = 60
    run_max_bytes: int = 12 * 1024 * 1024
    run_max_seconds: int = 20
    tunnel_max_bytes: int = 4 * 1024 * 1024
    tunnel_idle_seconds: int = 10
    max_pages: int = 2
    concurrency: int = 1

    def __post_init__(self) -> None:
        if (
            self.page_max_requests,
            self.page_max_bytes,
            self.page_max_seconds,
            self.run_max_requests,
            self.run_max_bytes,
            self.run_max_seconds,
            self.tunnel_max_bytes,
            self.tunnel_idle_seconds,
            self.max_pages,
            self.concurrency,
        ) != (40, 8 * 1024 * 1024, 15, 60, 12 * 1024 * 1024, 20, 4 * 1024 * 1024, 10, 2, 1):
            raise ValueError("browser resource limits are frozen")

    @staticmethod
    def resource_allowed(resource_type: str) -> bool:
        return resource_type.casefold() in ALLOWED_BROWSER_RESOURCE_TYPES


def browser_fallback_eligible(*, normalized_characters: int, non_boilerplate_blocks: int) -> bool:
    return normalized_characters < 400 or non_boilerplate_blocks < 2


def language_eligibility(*, sample: str, detected_code: str, confidence: float) -> tuple[bool, str | None]:
    if len(sample.encode("utf-8")) > 4096:
        sample = sample.encode("utf-8")[:4096].decode("utf-8", errors="ignore")
    if len(sample) < 200 or confidence < 0.80:
        return False, None
    if detected_code == "el":
        return True, "el-GR"
    if detected_code == "en":
        return True, "en-US"
    return False, None


def _parse_datetime(value: str) -> str | None:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _DateParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.article_depth = 0
        self.json_ld_depth = 0
        self.json_ld: list[str] = []
        self.candidates: list[tuple[DateProvenance, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value or "" for key, value in attrs}
        if tag == "article":
            self.article_depth += 1
        if tag == "script" and values.get("type", "").casefold() == "application/ld+json":
            self.json_ld_depth += 1
        if tag == "meta":
            key = (values.get("property") or values.get("name")).casefold()
            provenance = None
            if key == "article:published_time":
                provenance = DateProvenance.ARTICLE_PUBLISHED_TIME
            elif key in {"datepublished", "publishdate", "publication_date"}:
                provenance = DateProvenance.APPROVED_META
            if provenance and values.get("content"):
                self.candidates.append((provenance, values["content"]))
        if tag == "time" and self.article_depth and values.get("datetime"):
            self.candidates.append((DateProvenance.ARTICLE_TIME, values["datetime"]))

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self.article_depth:
            self.article_depth -= 1
        if tag == "script" and self.json_ld_depth:
            self.json_ld_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.json_ld_depth:
            self.json_ld.append(data)


def publication_date_from_html(value: str) -> tuple[str | None, DateProvenance]:
    parser = _DateParser()
    try:
        parser.feed(value)
        for block in parser.json_ld:
            parsed = json.loads(block)
            nodes = parsed if isinstance(parsed, list) else [parsed]
            for node in nodes:
                if isinstance(node, dict) and isinstance(node.get("datePublished"), str):
                    parser.candidates.append((DateProvenance.JSON_LD, node["datePublished"]))
    except (ValueError, json.JSONDecodeError):
        pass
    valid = [(kind, parsed) for kind, raw in parser.candidates if (parsed := _parse_datetime(raw))]
    dates = {item for _, item in valid}
    if not dates:
        return None, DateProvenance.NONE
    if len(dates) != 1:
        return None, DateProvenance.CONFLICT
    selected = next(iter(dates))
    precedence = (
        DateProvenance.JSON_LD,
        DateProvenance.ARTICLE_PUBLISHED_TIME,
        DateProvenance.APPROVED_META,
        DateProvenance.ARTICLE_TIME,
    )
    provenance = next(item for item in precedence if (item, selected) in valid)
    return selected, provenance
