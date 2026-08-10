"""Strict, explicit Orchestrator configuration."""

from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    database_path: Path
    voice_host: str = "127.0.0.1"
    voice_port: int = 8765
    control_host: str = "127.0.0.1"
    control_port: int = 8767
    control_bearer_token: str = ""
    inference_base_url: str = "http://127.0.0.1:8766/v1"
    inference_bearer_token: str = ""
    model_alias: str = "jarvis-local"
    model_timeout_seconds: float = 120.0
    default_speech_locale: str = "el-GR"
    active_budget_seconds: float = 120.0
    absolute_lifetime_seconds: float = 900.0
    approval_ttl_seconds: float = 600.0
    reconnect_ttl_seconds: float = 30.0
    delivery_ttl_seconds: float = 30.0
    model_request_limit: int = 4
    output_token_limit: int = 2048
    memory_enabled: bool = False
    memory_timeout_seconds: float = 0.25
    memory_recall_limit: int = 5
    memory_shadow_enabled: bool = False
    memory_shadow_timeout_seconds: float = 0.25

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OrchestratorConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown configuration field: {sorted(unknown)[0]}")
        data = dict(value)
        if "database_path" not in data:
            raise ValueError("database_path is required")
        data["database_path"] = Path(str(data["database_path"])).expanduser()
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "OrchestratorConfig":
        raw = Path(path).read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("configuration must be an object")
        return cls.from_mapping(parsed)

    def validate(self) -> None:
        if not self.database_path.is_absolute():
            raise ValueError("database_path must be absolute")
        for field in ("voice_host", "control_host"):
            if ipaddress.ip_address(getattr(self, field)) != ipaddress.ip_address("127.0.0.1"):
                raise ValueError(f"{field} must be 127.0.0.1")
        parsed = urlparse(self.inference_base_url)
        if parsed.scheme != "http" or parsed.hostname is None:
            raise ValueError("inference_base_url is invalid")
        if parsed.path.rstrip("/") != "/v1" or parsed.params or parsed.query or parsed.fragment or parsed.username is not None:
            raise ValueError("inference_base_url must be the fixed /v1 endpoint")
        if not ipaddress.ip_address(parsed.hostname).is_loopback:
            raise ValueError("inference_base_url must be loopback")
        if self.model_alias != "jarvis-local":
            raise ValueError("model_alias must be jarvis-local")
        if not self.control_bearer_token or len(self.control_bearer_token.encode()) > 512:
            raise ValueError("a bounded control bearer token is required")
        if not self.inference_bearer_token:
            raise ValueError("an inference bearer token is required")
        positive = (
            self.voice_port,
            self.control_port,
            self.model_timeout_seconds,
            self.active_budget_seconds,
            self.absolute_lifetime_seconds,
            self.approval_ttl_seconds,
            self.reconnect_ttl_seconds,
            self.delivery_ttl_seconds,
            self.model_request_limit,
            self.output_token_limit,
            self.memory_timeout_seconds,
            self.memory_recall_limit,
            self.memory_shadow_timeout_seconds,
        )
        if any(isinstance(item, bool) or item <= 0 for item in positive):
            raise ValueError("numeric configuration values must be positive")
        if not isinstance(self.memory_shadow_enabled, bool):
            raise ValueError("memory_shadow_enabled must be boolean")
        if not isinstance(self.memory_enabled, bool):
            raise ValueError("memory_enabled must be boolean")
        if (
            isinstance(self.memory_recall_limit, bool)
            or not isinstance(self.memory_recall_limit, int)
            or self.memory_recall_limit > 20
        ):
            raise ValueError("memory_recall_limit must be an integer from 1 to 20")
