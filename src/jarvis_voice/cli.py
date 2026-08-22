"""Source-run command line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from urllib.parse import urlparse

from .config import configured_voice_auth_key, load_voice_config
from .doctor import doctor_report, doctor_report_config
from .errors import BackendError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyrus-voice")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Run offline diagnostics")
    source = doctor.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--model-dir")
    doctor.add_argument("--active-cuda-probe", action="store_true", help="Explicitly load and exercise the configured local CUDA model")
    doctor.add_argument("--output", help="Write redacted doctor JSON atomically instead of stdout")
    macos_doctor = commands.add_parser("doctor-macos", help="Run offline macOS Apple Silicon diagnostics")
    macos_doctor.add_argument("--config", required=True)
    macos_doctor.add_argument("--output", help="Write redacted doctor JSON atomically instead of stdout")
    macos_cpu_doctor = commands.add_parser("doctor-macos-cpu", help="Run the explicit offline faster-whisper CPU-smoke diagnostic")
    macos_cpu_doctor.add_argument("--config", required=True)
    macos_cpu_doctor.add_argument("--model-dir", required=True)
    macos_cpu_doctor.add_argument("--output", help="Write redacted doctor JSON atomically instead of stdout")
    fake = commands.add_parser("fake-server", help="Run the loopback fake orchestrator")
    fake.add_argument("--config", required=True)
    auth_key = commands.add_parser(
        "create-auth-key",
        help="Create one external per-install Voice authentication key",
    )
    auth_key.add_argument("--output", dest="key_output", required=True)
    auth_key.add_argument("--key-version", type=int, default=1)
    commands.add_parser(
        "harden-private-state",
        help="Safely migrate Voice SQLite journals to the owner-only boundary",
    )
    run = commands.add_parser("run", help="Run the native Windows voice loop")
    run.add_argument("--config", required=True)
    run_macos = commands.add_parser("run-macos", help="Run the native macOS voice loop")
    run_macos.add_argument("--config", required=True)
    run_macos_cpu = commands.add_parser("run-macos-cpu", help="Run the explicit macOS faster-whisper CPU-smoke voice loop")
    run_macos_cpu.add_argument("--config", required=True)
    run_macos_cpu.add_argument("--model-dir", required=True)
    tray = commands.add_parser("tray", help="Run the native Windows tray application")
    tray.add_argument("--config")
    manifest = commands.add_parser("dist-manifest", help="Write an offline onedir integrity manifest")
    manifest.add_argument("--directory", required=True)
    verify = commands.add_parser("verify-dist", help="Verify an offline onedir integrity manifest")
    verify.add_argument("--directory", required=True)
    template = commands.add_parser("qualification-template", help="Write an empty strict native-smoke sample file")
    template.add_argument("--output", required=True)
    template.add_argument("--distribution", required=True)
    record = commands.add_parser("qualification-record", help="Append one content-free native-smoke sample")
    record.add_argument("--samples", required=True)
    record.add_argument("--case", required=True)
    record.add_argument("--outcome", required=True, choices=("pass", "fail"))
    record.add_argument("--metric", action="append", default=[])
    qualify = commands.add_parser("qualify-release", help="Create a strict native Windows release qualification report")
    qualify.add_argument("--config", required=True)
    qualify.add_argument("--distribution", required=True)
    qualify.add_argument("--samples", required=True)
    qualify.add_argument("--frozen-doctor", required=True)
    qualify.add_argument("--native-evidence", required=True)
    qualify.add_argument("--output", required=True)
    audit = commands.add_parser("artifact-audit", help="Discover already-local speech artifacts without loading them")
    audit.add_argument("--root", action="append", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--private-manifest", required=True)
    audit.add_argument("--max-depth", type=int, default=8)
    audit.add_argument("--max-directories", type=int, default=50_000)
    audit.add_argument("--max-candidates", type=int, default=100)
    audit.add_argument("--hash-candidates", action="store_true")
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _emit(value: object, output: str | None = None) -> None:
    if output is None:
        _print(value)
        return
    from .qualification import write_report
    write_report(output, value)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            if args.active_cuda_probe and not args.config:
                raise BackendError("cuda.active_probe_requires_config", "Active CUDA probe requires --config", subsystem="cuda")
            report = doctor_report_config(load_voice_config(args.config), active_cuda_probe=args.active_cuda_probe) if args.config else doctor_report(args.model_dir)
            _emit(report, args.output)
            return 0 if report["ready"] else 2
        if args.command == "doctor-macos":
            from .macos.doctor import macos_doctor_report
            report = macos_doctor_report(load_voice_config(args.config))
            _emit(report, args.output)
            return 0 if report["ready"] else 2
        if args.command == "doctor-macos-cpu":
            from .macos.asr_selection import with_faster_whisper_cpu_smoke
            from .macos.doctor import macos_doctor_report
            config = with_faster_whisper_cpu_smoke(load_voice_config(args.config), args.model_dir)
            report = macos_doctor_report(config)
            _emit(report, args.output)
            return 0 if report["ready"] else 2
        if args.command == "qualification-template":
            from .qualification import verified_distribution_manifest_sha256, write_sample_template
            write_sample_template(
                args.output,
                verified_distribution_manifest_sha256(args.distribution),
            )
            return 0
        if args.command == "qualification-record":
            from .qualification import parse_metrics, record_sample
            record_sample(args.samples, case=args.case, outcome=args.outcome, metrics=parse_metrics(args.metric))
            return 0
        if args.command == "dist-manifest":
            from .distribution import build_distribution_manifest, write_distribution_manifest
            output = write_distribution_manifest(args.directory)
            report = build_distribution_manifest(args.directory)
            _print({"ready": True, "code": "packaging.manifest_written", "files": len(report["files"]), "manifest_name": output.name, "network_used": False})
            return 0
        if args.command == "verify-dist":
            from .distribution import verify_distribution
            result = verify_distribution(args.directory)
            _print({**result.to_dict(), "ready": result.ok, "network_used": False})
            return 0 if result.ok else 2
        if args.command == "qualify-release":
            from .qualification import build_release_report, write_report
            report = build_release_report(
                load_voice_config(args.config),
                distribution=args.distribution,
                samples_path=args.samples,
                frozen_doctor_path=args.frozen_doctor,
                native_evidence_path=args.native_evidence,
            )
            write_report(args.output, report)
            return 0 if report["ready"] else 2
        if args.command == "artifact-audit":
            from .artifact_audit import run_artifact_audit
            report = run_artifact_audit(
                args.root,
                output=args.output,
                private_manifest=args.private_manifest,
                max_depth=args.max_depth,
                max_directories=args.max_directories,
                max_candidates=args.max_candidates,
                hash_candidates=args.hash_candidates,
            )
            return 0 if report["completed"] else 2
        if args.command == "create-auth-key":
            from .voice_auth import VoiceAuthError, create_voice_auth_key_file

            try:
                created = create_voice_auth_key_file(
                    args.key_output,
                    key_version=args.key_version,
                )
            except VoiceAuthError as exc:
                raise BackendError(
                    exc.code,
                    "Voice authentication key creation failed",
                    subsystem="voice_auth",
                ) from exc
            _print(
                {
                    "code": "voice_auth.key_created",
                    "key_version": created.key_version,
                    "ready": True,
                    "secret_disclosed": False,
                }
            )
            return 0
        if args.command == "harden-private-state":
            from .private_state import harden_voice_journal_state

            report = harden_voice_journal_state()
            _print(
                {
                    **report.to_dict(),
                    "content_captured": False,
                    "network_used": False,
                }
            )
            return 0 if report.ready else 2
        if args.command == "tray":
            from .tray_app import VoiceTrayApplication
            VoiceTrayApplication(args.config).run()
            return 0
        config = load_voice_config(args.config)
        if args.command == "fake-server":
            from .transport import LoopbackFakeServer
            parsed = urlparse(config.orchestrator_url)
            LoopbackFakeServer(
                auth_key=configured_voice_auth_key(config)
            ).run(parsed.hostname or "127.0.0.1", parsed.port or 8765)
            return 0
        if args.command == "run":
            from .runtime import WindowsVoiceRuntime
            WindowsVoiceRuntime(config).run()
            return 0
        if args.command == "run-macos":
            from .macos.runtime import MacVoiceRuntime
            MacVoiceRuntime(config).run()
            return 0
        if args.command == "run-macos-cpu":
            from .macos.asr_selection import with_faster_whisper_cpu_smoke
            from .macos.runtime import MacVoiceRuntime
            MacVoiceRuntime(with_faster_whisper_cpu_smoke(config, args.model_dir)).run()
            return 0
    except BackendError as exc:
        error_report = {"ready": False, "error": exc.to_log_record(), "network_used": False}
        _emit(error_report, getattr(args, "output", None))
        return 2
    raise AssertionError("argparse accepted an unknown command")
