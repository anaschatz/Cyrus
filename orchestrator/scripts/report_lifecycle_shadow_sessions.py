#!/usr/bin/env python3
"""Print a privacy-preserving report for a stopped JARVIS session database."""

from __future__ import annotations

import argparse
import json

from jarvis_orchestrator.shadow_audit import build_shadow_trial_report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="report-lifecycle-shadow-sessions")
    result.add_argument("--database", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parsed = parser().parse_args(argv)
    report = build_shadow_trial_report(parsed.database)
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 2 if report["decision"] == "shadow_trial_rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
