"""Live, fail-closed qualification gate for SearXNG and direct fetch."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from jarvis_orchestrator.domain import canonical_json

from .api import parse_research_request
from .config import QUALIFICATION_FIELDS, load_json_object

QUALIFICATION_SCHEMA_VERSION = 2
MANDATORY_LIVE_CASES = frozenset(
    {
        "robots_allow", "robots_disallow", "robots_404", "robots_410", "robots_fail_closed",
        "redirect_revalidation", "ipv4_private_reserved", "ipv6_private_reserved", "mixed_dns_answers",
        "dns_rebinding", "mime_limit", "compressed_body_limit", "decompressed_body_limit",
        "single_transient_retry", "deadline", "cancellation",
    }
)
_SECRET = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b|\b(?:sk|pk|api)[-_][A-Za-z0-9_-]{20,}\b)"
)
_SUBPROCESS_ENV_KEYS = frozenset(
    {
        "DOCKER_CONFIG", "DOCKER_CONTEXT", "DOCKER_HOST",
        "JARVIS_RESEARCH_IMAGE", "JARVIS_EGRESS_GUARD_IMAGE", "SEARXNG_IMAGE",
        "JARVIS_PYTHON_IMAGE", "JARVIS_RESEARCH_BEARER_TOKEN_SOURCE",
        "JARVIS_EGRESS_PROXY_CREDENTIALS_SOURCE", "JARVIS_RESEARCH_QUALIFICATION_FILE",
        "JARVIS_QUALIFICATION_SEARXNG_SETTINGS_SOURCE",
    }
)


@dataclass(frozen=True, slots=True)
class CaseResultV1:
    case_id: str
    outcome: str
    code: str
    duration_ms: int
    response_sha256: str


def _case_dict(item: CaseResultV1) -> dict[str, object]:
    return {
        "case_id": item.case_id,
        "outcome": item.outcome,
        "code": item.code,
        "duration_ms": item.duration_ms,
        "response_sha256": item.response_sha256,
    }


def qualification_digest(evidence: dict[str, object]) -> str:
    """Hash stable gate decisions without the self-referential digest or timings.

    The full record has a second digest.  This digest is the value bound into the
    eight-field Research API identity and therefore deliberately excludes its
    own value, live durations and raw response hashes.  Artifact/image digests
    and every mandatory pass/fail decision remain covered.
    """

    identity = evidence.get("qualification_identity")
    cases = evidence.get("cases")
    if not isinstance(identity, dict) or not isinstance(cases, list):
        raise ValueError("qualification evidence is incomplete")
    identity_without_digest = {
        key: value for key, value in identity.items() if key != "qualification_sha256"
    }
    stable_cases: list[dict[str, object]] = []
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("qualification case is invalid")
        if item.get("case_id") == "digest_binding":
            continue
        stable_cases.append(
            {
                "case_id": item.get("case_id"),
                "outcome": item.get("outcome"),
                "code": item.get("code"),
            }
        )
    payload = {
        "schema_version": evidence.get("schema_version"),
        "browser_enabled": evidence.get("browser_enabled"),
        "qualification_identity": identity_without_digest,
        "artifact_sha256": evidence.get("artifact_sha256"),
        "image_sha256": evidence.get("image_sha256"),
        "cases": stable_cases,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def record_digest(evidence: dict[str, object]) -> str:
    """Hash the complete canonical evidence record except this digest field."""

    payload = {key: value for key, value in evidence.items() if key != "record_sha256"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def verify_accepted_evidence(evidence: dict[str, object]) -> None:
    """Fail closed unless an evidence record is canonical, complete and bound."""

    required = {
        "schema_version", "gate_status", "browser_enabled", "qualification_identity",
        "artifact_sha256", "image_sha256", "cases", "evidence_sha256", "record_sha256",
    }
    if set(evidence) != required or evidence.get("schema_version") != QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("qualification evidence fields are invalid")
    if evidence.get("gate_status") != "ACCEPTED" or evidence.get("browser_enabled") is not False:
        raise ValueError("qualification gate is not accepted")
    identity = evidence.get("qualification_identity")
    cases = evidence.get("cases")
    if not isinstance(identity, dict) or set(identity) != QUALIFICATION_FIELDS or not isinstance(cases, list):
        raise ValueError("qualification evidence identity is invalid")
    if not cases or any(
        not isinstance(item, dict)
        or set(item) != {"case_id", "outcome", "code", "duration_ms", "response_sha256"}
        or item.get("outcome") != "PASS"
        for item in cases
    ):
        raise ValueError("qualification evidence contains a failed case")
    digest = qualification_digest(evidence)
    if evidence.get("evidence_sha256") != digest or identity.get("qualification_sha256") != digest:
        raise ValueError("qualification evidence digest binding is invalid")
    if evidence.get("record_sha256") != record_digest(evidence):
        raise ValueError("qualification evidence record digest is invalid")


def _run(command: list[str], *, cwd: Path, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    executable = shutil.which(command[0], path=os.environ.get("PATH"))
    if executable is None:
        return subprocess.CompletedProcess(command, 127, "", "command_unavailable")
    environment = {"PATH": os.path.dirname(executable) + ":/usr/local/bin:/usr/bin:/bin"}
    environment.update(
        (key, os.environ[key]) for key in _SUBPROCESS_ENV_KEYS if key in os.environ
    )
    return subprocess.run(
        [executable, *command[1:]], cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False,
        env=environment,
    )


def _compose_command() -> list[str]:
    standalone = shutil.which("docker-compose", path=os.environ.get("PATH"))
    if standalone:
        return [standalone]
    docker = shutil.which("docker", path=os.environ.get("PATH"))
    if docker:
        bundled = Path(docker).resolve().parent.parent / "cli-plugins" / "docker-compose"
        if bundled.is_file() and os.access(bundled, os.X_OK):
            return [str(bundled)]
    return ["docker", "compose"]


class LiveQualificationRunnerV1:
    def __init__(
        self,
        manifest: dict[str, object],
        *,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
    ) -> None:
        required = {
            "schema_version", "api_base_url", "bearer_token_file", "compose_directory", "compose_files",
            "request_template", "cases", "forbidden_log_markers", "artifact_files",
        }
        if set(manifest) != required:
            raise ValueError("qualification manifest fields are invalid")
        if manifest["schema_version"] != 1 or not isinstance(manifest["request_template"], dict):
            raise ValueError("qualification manifest version is invalid")
        if not all(isinstance(manifest[name], list) for name in ("cases", "forbidden_log_markers", "compose_files")) or not isinstance(manifest["artifact_files"], dict):
            raise ValueError("qualification manifest collections are invalid")
        cases = manifest["cases"]
        ids = {item.get("case_id") for item in cases if isinstance(item, dict)}
        if ids != MANDATORY_LIVE_CASES or len(cases) != len(MANDATORY_LIVE_CASES):
            raise ValueError("qualification manifest mandatory cases are incomplete")
        self._manifest = manifest
        self._run = command_runner
        self._images: dict[str, str] = {}
        self._compose = _compose_command()
        for filename in manifest["compose_files"]:  # type: ignore[union-attr]
            if not isinstance(filename, str) or "/" in filename or "\\" in filename or not filename.endswith((".yml", ".yaml")):
                raise ValueError("qualification compose filename is invalid")
            self._compose.extend(("-f", filename))

    async def run(self) -> dict[str, object]:
        results: list[CaseResultV1] = []
        qualification: dict[str, object] = {}
        base = str(self._manifest["api_base_url"]).rstrip("/")
        if not base.startswith(("http://127.0.0.1:", "http://[::1]:")) or not base.endswith("/v1"):
            raise ValueError("qualification API must be loopback /v1")
        token = Path(str(self._manifest["bearer_token_file"])).read_text(encoding="utf-8").strip()
        headers = {"Authorization": "Bearer " + token}
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False, timeout=50.0) as client:
            results.append(await self._auth_case(client, base, headers))
            try:
                response = await client.get(base + "/qualification", headers=headers)
                value = response.json()
                if response.status_code != 200 or not isinstance(value, dict) or set(value) != QUALIFICATION_FIELDS:
                    raise ValueError
                qualification = value
            except (httpx.HTTPError, ValueError, json.JSONDecodeError):
                results.append(CaseResultV1("qualification_identity", "FAIL", "identity_invalid", 0, ""))
            else:
                results.append(CaseResultV1("qualification_identity", "PASS", "ok", 0, hashlib.sha256(response.content).hexdigest()))
            for case in sorted(self._manifest["cases"], key=lambda item: item["case_id"]):  # type: ignore[index,union-attr]
                results.append(await self._network_case(client, base, headers, case))  # type: ignore[arg-type]
        results.extend(self._topology_cases(Path(str(self._manifest["compose_directory"]))))
        results.append(self._sanitized_logs(Path(str(self._manifest["compose_directory"]))))
        browser_disabled = qualification.get("browser_qualification_sha256") == ""
        results.append(CaseResultV1("browser_disabled", "PASS" if browser_disabled else "FAIL", "ok" if browser_disabled else "browser_digest_present", 0, ""))
        artifacts = self._artifact_digests(Path(str(self._manifest["compose_directory"])))
        base_cases = sorted(results, key=lambda item: item.case_id)
        body: dict[str, object] = {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "gate_status": "REJECTED",
            "browser_enabled": False,
            "qualification_identity": qualification,
            "artifact_sha256": artifacts,
            "image_sha256": dict(sorted(self._images.items())),
            "cases": [_case_dict(item) for item in base_cases],
        }
        digest = qualification_digest(body)
        binding = qualification.get("qualification_sha256") == digest
        base_cases.append(
            CaseResultV1(
                "digest_binding", "PASS" if binding else "FAIL",
                "ok" if binding else "qualification_digest_mismatch", 0, "",
            )
        )
        body["cases"] = [_case_dict(item) for item in sorted(base_cases, key=lambda item: item.case_id)]
        body["gate_status"] = (
            "ACCEPTED" if all(item.outcome == "PASS" for item in base_cases) else "REJECTED"
        )
        bound = {**body, "evidence_sha256": digest}
        return {**bound, "record_sha256": record_digest(bound)}

    async def _auth_case(self, client: httpx.AsyncClient, base: str, headers: dict[str, str]) -> CaseResultV1:
        started = time.monotonic()
        try:
            denied, allowed = await asyncio.gather(
                client.get(base + "/qualification", headers={"Authorization": "Bearer invalid"}),
                client.get(base + "/qualification", headers=headers),
            )
            passed = denied.status_code == 401 and allowed.status_code == 200
            raw = denied.content + b"\n" + allowed.content
            return CaseResultV1("authenticated_api", "PASS" if passed else "FAIL", "ok" if passed else "auth_boundary_failed", int((time.monotonic()-started)*1000), hashlib.sha256(raw).hexdigest())
        except httpx.HTTPError:
            return CaseResultV1("authenticated_api", "FAIL", "api_unavailable", int((time.monotonic()-started)*1000), "")

    async def _network_case(
        self, client: httpx.AsyncClient, base: str, headers: dict[str, str], case: dict[str, object]
    ) -> CaseResultV1:
        expected = {"case_id", "idempotency_key", "query", "expected_http", "expected_source_count", "cancel_after_ms"}
        case_id = str(case.get("case_id", "invalid"))
        started = time.monotonic()
        if set(case) != expected or not isinstance(case.get("query"), str):
            return CaseResultV1(case_id, "FAIL", "case_schema_invalid", 0, "")
        request = deepcopy(self._manifest["request_template"])
        try:
            request["plan"]["canonical_query"] = case["query"]  # type: ignore[index]
            request["plan"]["subqueries"] = [case["query"]]  # type: ignore[index]
        except (KeyError, TypeError):
            return CaseResultV1(case_id, "FAIL", "request_template_invalid", 0, "")
        raw = canonical_json(request).encode("utf-8")
        try:
            parse_research_request(raw)
        except Exception:
            return CaseResultV1(case_id, "FAIL", "request_contract_invalid", 0, "")
        outgoing = {**headers, "Content-Type": "application/json", "Idempotency-Key": str(case["idempotency_key"])}
        try:
            task = asyncio.create_task(client.post(base + "/research", headers=outgoing, content=raw))
            cancel_ms = int(case["cancel_after_ms"])
            if cancel_ms:
                await asyncio.sleep(cancel_ms / 1000)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                passed = case_id == "cancellation"
                return CaseResultV1(case_id, "PASS" if passed else "FAIL", "cancelled", int((time.monotonic()-started)*1000), "")
            response = await task
            source_count = -1
            if response.status_code == 200:
                parsed = response.json()
                source_count = len(parsed.get("corpus", {}).get("sources", []))
            passed = response.status_code == int(case["expected_http"]) and source_count == int(case["expected_source_count"])
            return CaseResultV1(case_id, "PASS" if passed else "FAIL", "ok" if passed else "unexpected_result", int((time.monotonic()-started)*1000), hashlib.sha256(response.content).hexdigest())
        except (httpx.HTTPError, ValueError, json.JSONDecodeError):
            return CaseResultV1(case_id, "FAIL", "network_case_failed", int((time.monotonic()-started)*1000), "")

    def _topology_cases(self, cwd: Path) -> list[CaseResultV1]:
        results: list[CaseResultV1] = []
        searx_id = self._run(self._compose + ["ps", "-q", "searxng"], cwd=cwd)
        research_id = self._run(self._compose + ["ps", "-q", "research-api"], cwd=cwd)
        guard_id = self._run(self._compose + ["ps", "-q", "egress-guard"], cwd=cwd)
        if (
            searx_id.returncode or research_id.returncode or guard_id.returncode
            or not searx_id.stdout.strip() or not research_id.stdout.strip() or not guard_id.stdout.strip()
        ):
            return [
                CaseResultV1("searxng_isolation", "FAIL", "containers_unavailable", 0, ""),
                CaseResultV1("no_direct_route", "FAIL", "containers_unavailable", 0, ""),
                CaseResultV1("late_completion_no_persistence", "FAIL", "containers_unavailable", 0, ""),
                CaseResultV1("egress_nonroot_readonly", "FAIL", "containers_unavailable", 0, ""),
                CaseResultV1("image_digests", "FAIL", "containers_unavailable", 0, ""),
            ]
        inspected = self._run(["docker", "inspect", searx_id.stdout.strip()], cwd=cwd)
        try:
            info = json.loads(inspected.stdout)[0]
            ports = info["NetworkSettings"]["Ports"]
            isolated = not any(value for value in ports.values())
            if isinstance(info.get("Image"), str) and info["Image"].startswith("sha256:"):
                self._images["searxng"] = info["Image"].removeprefix("sha256:")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            isolated = False
        results.append(CaseResultV1("searxng_isolation", "PASS" if isolated else "FAIL", "ok" if isolated else "published_port_present", 0, hashlib.sha256(inspected.stdout.encode()).hexdigest() if inspected.stdout else ""))
        direct = self._run(self._compose + ["exec", "-T", "research-api", "python", "-m", "jarvis_research.runtime_probe", "direct-route-blocked"], cwd=cwd)
        results.append(CaseResultV1("no_direct_route", "PASS" if direct.returncode == 0 else "FAIL", "ok" if direct.returncode == 0 else "direct_route_present", 0, ""))
        research_inspect = self._run(["docker", "inspect", research_id.stdout.strip()], cwd=cwd)
        try:
            research_info = json.loads(research_inspect.stdout)[0]
            mounts = research_info["Mounts"]
            durable = any(item.get("Type") == "volume" and item.get("RW") for item in mounts)
            if isinstance(research_info.get("Image"), str) and research_info["Image"].startswith("sha256:"):
                self._images["research-api"] = research_info["Image"].removeprefix("sha256:")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            durable = True
        results.append(CaseResultV1("late_completion_no_persistence", "PASS" if not durable else "FAIL", "ok" if not durable else "writable_volume_present", 0, hashlib.sha256(research_inspect.stdout.encode()).hexdigest() if research_inspect.stdout else ""))
        guard_inspect = self._run(["docker", "inspect", guard_id.stdout.strip()], cwd=cwd)
        try:
            guard = json.loads(guard_inspect.stdout)[0]
            secure = bool(guard["HostConfig"]["ReadonlyRootfs"]) and guard["Config"]["User"] not in {"", "0", "root"}
            if isinstance(guard.get("Image"), str) and guard["Image"].startswith("sha256:"):
                self._images["egress-guard"] = guard["Image"].removeprefix("sha256:")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            secure = False
        results.append(CaseResultV1("egress_nonroot_readonly", "PASS" if secure else "FAIL", "ok" if secure else "container_security_invalid", 0, hashlib.sha256(guard_inspect.stdout.encode()).hexdigest() if guard_inspect.stdout else ""))
        images_valid = set(self._images) == {"research-api", "searxng", "egress-guard"} and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in self._images.values())
        results.append(CaseResultV1("image_digests", "PASS" if images_valid else "FAIL", "ok" if images_valid else "image_digest_missing", 0, ""))
        return results

    def _sanitized_logs(self, cwd: Path) -> CaseResultV1:
        result = self._run(
            self._compose + ["logs", "--no-color", "research-api", "egress-guard", "searxng", "searx-egress-auth"],
            cwd=cwd,
        )
        markers = self._manifest["forbidden_log_markers"]
        safe_markers = isinstance(markers, list) and all(isinstance(item, str) and item and item not in result.stdout for item in markers)
        passed = result.returncode == 0 and safe_markers and _SECRET.search(result.stdout) is None
        return CaseResultV1("sanitized_logs", "PASS" if passed else "FAIL", "ok" if passed else "sensitive_log_content", 0, hashlib.sha256(result.stdout.encode()).hexdigest() if result.stdout else "")

    def _artifact_digests(self, cwd: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, relative in sorted(self._manifest["artifact_files"].items()):  # type: ignore[union-attr]
            if not isinstance(name, str) or not isinstance(relative, str) or "/" in name:
                raise ValueError("artifact manifest is invalid")
            path = (cwd / relative).resolve()
            if cwd.resolve() not in path.parents or not path.is_file():
                raise ValueError("artifact path is invalid")
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis-research-qualify")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = load_json_object(args.manifest)
        evidence = asyncio.run(LiveQualificationRunnerV1(manifest).run())
        args.output.write_text(canonical_json(evidence) + "\n", encoding="utf-8")
    except Exception as exc:
        evidence = {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "gate_status": "REJECTED",
            "browser_enabled": False,
            "cases": [],
            "error_code": type(exc).__name__,
        }
        digest = hashlib.sha256(canonical_json(evidence).encode("utf-8")).hexdigest()
        bound = {**evidence, "evidence_sha256": digest}
        args.output.write_text(canonical_json({**bound, "record_sha256": record_digest(bound)}) + "\n", encoding="utf-8")
        return 2
    return 0 if evidence["gate_status"] == "ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
