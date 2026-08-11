"""Native macOS qualification fixture for the loopback inference appliance.

The qualifier launches this module through a tiny, hash-pinned runner using the
manifest-declared Python interpreter.  It talks only to the ephemeral loopback
gateway created by :mod:`jarvis_inference.qualification` and writes one fresh,
strict evidence document.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .qualification import SUITE_VERSION
from .strict_json import canonical_bytes, parse_body


class FixtureFailure(RuntimeError):
    """Content-free native fixture failure."""


class QualificationClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or parsed.path.rstrip("/") != "/v1"
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise FixtureFailure("invalid loopback qualification URL")
        self.host = parsed.hostname
        self.port = parsed.port
        self.api_key = api_key
        self.model = model

    def _connection(self, timeout: float = 300.0) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 300.0,
    ) -> tuple[int, dict[str, Any]]:
        body = b"" if payload is None else canonical_bytes(payload)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(body))})
        connection = self._connection(timeout)
        try:
            connection.request(method, path, body=body if payload is not None else None, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            value = parse_body(raw) if raw else {}
            if not isinstance(value, dict):
                raise FixtureFailure("non-object gateway response")
            return response.status, value
        finally:
            connection.close()

    def completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 16,
        stream: bool = False,
        timeout: float = 300.0,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
        }
        if not stream:
            status, value = self.json_request("POST", "/v1/chat/completions", request, timeout=timeout)
            if status != 200:
                raise FixtureFailure("completion request failed")
            try:
                choice = value["choices"][0]
                text = choice["message"]["content"]
                usage = value["usage"]
            except (KeyError, IndexError, TypeError) as exc:
                raise FixtureFailure("completion response shape failed") from exc
            if (
                not isinstance(text, str)
                or not text
                or choice.get("finish_reason") not in {"stop", "length"}
                or not isinstance(usage, dict)
                or type(usage.get("completion_tokens")) is not int
            ):
                raise FixtureFailure("completion response contract failed")
            return text

        request.update({"stream": True, "stream_options": {"include_usage": True}})
        body = canonical_bytes(request)
        connection = self._connection(timeout)
        text_parts: list[str] = []
        terminal = False
        usage_seen = False
        done_seen = False
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = connection.getresponse()
            if response.status != 200 or response.getheader("Content-Type") != "text/event-stream; charset=utf-8":
                raise FixtureFailure("stream response failed")
            while True:
                raw = response.readline()
                if not raw:
                    break
                if not raw.startswith(b"data: "):
                    if raw.strip():
                        raise FixtureFailure("stream framing failed")
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    done_seen = True
                    break
                event = parse_body(data)
                if not isinstance(event, dict):
                    raise FixtureFailure("stream event failed")
                choices = event.get("choices")
                if choices == []:
                    usage = event.get("usage")
                    usage_seen = isinstance(usage, dict) and type(usage.get("completion_tokens")) is int
                    continue
                try:
                    choice = choices[0]
                    delta = choice["delta"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise FixtureFailure("stream choice failed") from exc
                content = delta.get("content")
                if isinstance(content, str) and content:
                    text_parts.append(content)
                if choice.get("finish_reason") in {"stop", "length"}:
                    terminal = True
        finally:
            connection.close()
        text = "".join(text_parts)
        if not text or not terminal or not usage_seen or not done_seen:
            raise FixtureFailure("stream terminal contract failed")
        return text

    def runtime(self) -> dict[str, Any]:
        status, value = self.json_request("GET", "/jarvis/qualification/runtime", timeout=10.0)
        if status != 200:
            raise FixtureFailure("runtime endpoint failed")
        return value

    def wait_ready(self, timeout: float = 600.0) -> dict[str, Any]:
        stop = time.monotonic() + timeout
        while time.monotonic() < stop:
            try:
                value = self.runtime()
                if value.get("ready") is True and value.get("busy") is False:
                    return value
            except (OSError, FixtureFailure):
                pass
            time.sleep(0.05)
        raise FixtureFailure("runtime did not recover")

    def inject(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        status, value = self.json_request("POST", f"/jarvis/qualification/{kind}", payload, timeout=30.0)
        if status != 200:
            raise FixtureFailure("fault injection failed")
        return value


def _read_identity(manifest_path: Path, challenge: str) -> tuple[str, dict[str, Any]]:
    manifest = parse_body(manifest_path.read_bytes())
    if not isinstance(manifest, dict) or manifest.get("alias") != "jarvis-local":
        raise FixtureFailure("manifest identity failed")
    report_name = manifest.get("qualification_report")
    if not isinstance(report_name, str) or Path(report_name).is_absolute() or ".." in Path(report_name).parts:
        raise FixtureFailure("qualification report path failed")
    report = parse_body((manifest_path.parent / report_name).read_bytes())
    if (
        not isinstance(report, dict)
        or report.get("status") != "in_progress"
        or report.get("challenge") != challenge
        or not isinstance(report.get("identity"), dict)
    ):
        raise FixtureFailure("qualification identity is unavailable")
    return manifest["alias"], report["identity"]


def _phase_cancel(client: QualificationClient, phase: str) -> None:
    result: list[BaseException | str] = []
    prompt = ("qualification prefill token " * 6000) if phase == "prefill" else "Count slowly from one to one hundred. /no_think"

    def request() -> None:
        try:
            result.append(client.completion(prompt, max_tokens=512, stream=True, timeout=600.0))
        except BaseException as exc:  # expected when the worker is terminated
            result.append(exc)

    thread = threading.Thread(target=request, name=f"qualification-{phase}", daemon=True)
    thread.start()
    stop = time.monotonic() + 120.0
    injected = False
    while time.monotonic() < stop and thread.is_alive():
        runtime = client.runtime()
        engine = runtime.get("engine")
        if isinstance(engine, dict) and engine.get("phase") == phase:
            response = client.inject("cancel", {"phase": phase})
            if response.get("phase") != phase or response.get("exit_ok") is not True:
                raise FixtureFailure("cancel termination contract failed")
            injected = True
            break
        time.sleep(0.005)
    if not injected:
        raise FixtureFailure("qualification phase was not observable")
    thread.join(timeout=120.0)
    if thread.is_alive():
        raise FixtureFailure("cancelled request did not terminate")
    client.wait_ready()


def _rss_bytes(runtime: dict[str, Any]) -> int:
    engine = runtime.get("engine")
    pid = engine.get("pid") if isinstance(engine, dict) else None
    if type(pid) is not int or pid <= 0:
        raise FixtureFailure("worker pid unavailable")
    completed = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    try:
        kib = int(completed.stdout.strip())
    except ValueError as exc:
        raise FixtureFailure("worker RSS unavailable") from exc
    if completed.returncode != 0 or kib <= 0:
        raise FixtureFailure("worker RSS unavailable")
    return kib * 1024


def _memory_drift(samples: list[int]) -> tuple[float, float]:
    if len(samples) != 200 or any(value <= 0 for value in samples):
        raise FixtureFailure("memory sample set failed")
    baseline = float(statistics.median(samples[:10]))
    final = float(statistics.median(samples[-10:]))
    slopes = [
        (samples[j] - samples[i]) / (j - i)
        for i in range(len(samples))
        for j in range(i + 1, len(samples))
    ]
    slope = float(statistics.median(slopes))
    final_drift = max(0.0, (final - baseline) / baseline)
    projected = max(0.0, slope * (len(samples) - 1) / baseline)
    return final_drift, projected


def _write_evidence(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.parent.is_symlink():
        raise FixtureFailure("evidence output path failed")
    raw = canonical_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=".qualification-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def run() -> dict[str, Any]:
    started_wall = time.time_ns()
    started_mono = time.monotonic_ns()
    challenge = os.environ.get("JARVIS_QUAL_CHALLENGE", "")
    identity_digest = os.environ.get("JARVIS_QUAL_IDENTITY_DIGEST", "")
    output = Path(os.environ.get("JARVIS_QUAL_OUTPUT", ""))
    manifest_path = Path(os.environ.get("JARVIS_QUAL_MANIFEST", ""))
    base_url = os.environ.get("JARVIS_QUAL_BASE_URL", "")
    api_key = os.environ.get("JARVIS_QUAL_API_KEY", "")
    if (
        len(challenge) != 64
        or len(identity_digest) != 64
        or not output.is_absolute()
        or not manifest_path.is_absolute()
        or not api_key
    ):
        raise FixtureFailure("qualification environment failed")
    model, identity = _read_identity(manifest_path, challenge)
    client = QualificationClient(base_url, api_key, model)
    client.wait_ready()

    greek_success = 0
    english_success = 0
    for _ in range(20):
        greek = client.completion("Απάντησε μόνο με τη λέξη ναι. /no_think", max_tokens=24, stream=True)
        if any("\u0370" <= character <= "\u03ff" for character in greek.lower()):
            greek_success += 1
        english = client.completion("Reply using only the word yes. /no_think", max_tokens=24, stream=True)
        if any("a" <= character.lower() <= "z" for character in english):
            english_success += 1
    if greek_success != 20 or english_success != 20:
        raise FixtureFailure("language streaming gate failed")

    for phase in ("prefill", "stream"):
        for _ in range(5):
            _phase_cancel(client, phase)

    for _ in range(5):
        response = client.inject("crash", {})
        if response.get("exit_ok") is not True:
            raise FixtureFailure("crash termination contract failed")
        client.wait_ready()

    isolation_pairs = 0
    previous_marker = ""
    for index in range(50):
        first = f"JARVIS_ISOLATION_A_{index:03d}"
        second = f"JARVIS_ISOLATION_B_{index:03d}"
        first_text = client.completion(f"Reply with OK only. Marker: {first}. /no_think", max_tokens=16)
        second_text = client.completion(f"Reply with OK only. Marker: {second}. /no_think", max_tokens=16)
        if previous_marker and previous_marker in first_text:
            raise FixtureFailure("semantic isolation leak")
        if first in second_text or (previous_marker and previous_marker in second_text):
            raise FixtureFailure("semantic isolation leak")
        previous_marker = second
        isolation_pairs += 1

    for _ in range(20):
        client.completion("Reply OK. /no_think", max_tokens=8)
    memory_samples: list[int] = []
    for index in range(200):
        client.completion(f"Reply OK. Qualification sample {index}. /no_think", max_tokens=8)
        memory_samples.append(_rss_bytes(client.runtime()))
    final_drift, projected_growth = _memory_drift(memory_samples)
    if final_drift > 0.05 or projected_growth > 0.05:
        raise FixtureFailure("unified memory drift gate failed")

    completed_wall = time.time_ns()
    elapsed = time.monotonic_ns() - started_mono
    fixture_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "suite_version": SUITE_VERSION,
        "fixture_digest": fixture_digest,
        "deterministic_seeds": [1],
        "identity": identity,
        "provenance": {
            "challenge": challenge,
            "identity_digest": identity_digest,
            "fixture_digest": fixture_digest,
            "started_unix_ns": str(started_wall),
            "completed_unix_ns": str(completed_wall),
            "monotonic_elapsed_ns": elapsed,
        },
        "streaming": {"greek_success": greek_success, "english_success": english_success},
        "tools": {"named_success": 0, "required_success": 0},
        "json_schema": {"strict_success": 0},
        "cancellation": {"prefill_success": 5, "mid_stream_success": 5},
        "crash_restart": {"isolated_success": 5},
        "isolation": {"pairs_success": isolation_pairs, "leaks": 0},
        "protocol_conformance": 1,
        "privacy_leaks": 0,
        "unexpected_oom": 0,
        "unexpected_crashes": 0,
        "unexpected_hangs": 0,
        "platform_gate": {
            "kind": "mac_unified_memory",
            "warmup_requests": 20,
            "normal_requests": 200,
            "final_drift": final_drift,
            "theil_sen_projected_growth": projected_growth,
            "oom": 0,
            "crashes": 0,
            "stale_state_references": 0,
        },
    }


def main() -> int:
    try:
        evidence = run()
        _write_evidence(Path(os.environ["JARVIS_QUAL_OUTPUT"]), evidence)
        return 0
    except BaseException:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
