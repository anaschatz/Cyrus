"""Strict runtime configuration for the isolated Research API."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

QUALIFICATION_FIELDS = frozenset(
    {
        "contract_version",
        "model_sha256",
        "profile_sha256",
        "qualification_sha256",
        "language_detector_version",
        "language_detector_sha256",
        "authority_registry_version",
        "browser_qualification_sha256",
    }
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
ASSET_ROOT = Path(__file__).with_name("assets")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("configuration has duplicate fields")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_pairs)
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")
    return value


@dataclass(frozen=True, slots=True)
class QualificationIdentityV1:
    contract_version: str
    model_sha256: str
    profile_sha256: str
    qualification_sha256: str
    language_detector_version: str
    language_detector_sha256: str
    authority_registry_version: str
    browser_qualification_sha256: str

    @classmethod
    def load(cls, path: Path) -> "QualificationIdentityV1":
        raw = load_json_object(path)
        if set(raw) != QUALIFICATION_FIELDS or not all(isinstance(value, str) for value in raw.values()):
            raise ValueError("qualification configuration fields are invalid")
        result = cls(**raw)  # type: ignore[arg-type]
        if result.contract_version != "research-corpus/1":
            raise ValueError("qualification contract version is invalid")
        for name in ("model_sha256", "profile_sha256", "qualification_sha256", "language_detector_sha256"):
            if _HASH.fullmatch(getattr(result, name)) is None:
                raise ValueError(f"{name} is invalid")
        if result.browser_qualification_sha256 != "":
            raise ValueError("browser qualification must remain disabled")
        detector = load_json_object(ASSET_ROOT / "language-detector-v1.json")
        registry = load_json_object(ASSET_ROOT / "authority-registry-v1.json")
        if result.language_detector_version != detector.get("version"):
            raise ValueError("language detector version mismatch")
        if result.language_detector_sha256 != file_sha256(ASSET_ROOT / "language-detector-v1.json"):
            raise ValueError("language detector digest mismatch")
        if result.authority_registry_version != registry.get("version"):
            raise ValueError("authority registry version mismatch")
        return result

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in sorted(QUALIFICATION_FIELDS)}


@dataclass(frozen=True, slots=True)
class RuntimeConfigV1:
    bind_host: str
    bind_port: int
    bearer_token: str
    proxy_credentials: str
    qualification: QualificationIdentityV1
    searxng_url: str = "http://searxng:8080/search"
    egress_proxy_url: str = "http://egress-guard:8080"

    @classmethod
    def from_environment(cls, env: dict[str, str] | None = None) -> "RuntimeConfigV1":
        values = dict(os.environ if env is None else env)
        bind = values.get("JARVIS_RESEARCH_BIND", "127.0.0.1:8770")
        try:
            host, raw_port = bind.rsplit(":", 1)
            port = int(raw_port)
        except (ValueError, TypeError) as exc:
            raise ValueError("research bind is invalid") from exc
        if host not in {"127.0.0.1", "::1", "0.0.0.0"} or not 1 <= port <= 65535:
            raise ValueError("research bind is invalid")
        token_path = Path(values.get("JARVIS_RESEARCH_BEARER_TOKEN_FILE", "/run/secrets/research_bearer_token"))
        proxy_path = Path(values.get("JARVIS_EGRESS_PROXY_CREDENTIALS_FILE", "/run/secrets/egress_proxy_credentials"))
        identity_path = Path(values.get("JARVIS_RESEARCH_QUALIFICATION_FILE", "/run/config/research-qualification.json"))
        token = token_path.read_text(encoding="utf-8").strip()
        proxy_credentials = proxy_path.read_text(encoding="utf-8").strip()
        if not 32 <= len(token.encode("utf-8")) <= 512 or "\n" in token or "\r" in token:
            raise ValueError("research bearer token is invalid")
        if not 32 <= len(proxy_credentials.encode("utf-8")) <= 512 or "\n" in proxy_credentials or "\r" in proxy_credentials:
            raise ValueError("egress proxy credentials are invalid")
        searxng = values.get("JARVIS_RESEARCH_SEARXNG_URL", "http://searxng:8080/search")
        proxy = values.get("JARVIS_RESEARCH_EGRESS_PROXY_URL", "http://egress-guard:8080")
        searx_target = (urlsplit(searxng).scheme, urlsplit(searxng).hostname, urlsplit(searxng).port, urlsplit(searxng).path)
        allowed_searx = {("http", "searxng", 8080, "/search")}
        if values.get("JARVIS_RESEARCH_QUALIFICATION_MODE") == "1":
            allowed_searx.add(("http", "qualification-search", 8080, "/search"))
        if searx_target not in allowed_searx:
            raise ValueError("SearXNG endpoint is invalid")
        if (urlsplit(proxy).scheme, urlsplit(proxy).hostname, urlsplit(proxy).port) != ("http", "egress-guard", 8080):
            raise ValueError("egress proxy endpoint is invalid")
        return cls(host, port, token, proxy_credentials, QualificationIdentityV1.load(identity_path), searxng, proxy)
