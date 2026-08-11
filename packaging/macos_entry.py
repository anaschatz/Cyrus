"""Signed macOS app entry point with external config/model state."""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

from jarvis_voice.config import load_voice_config
from jarvis_voice.macos.package_profile import apply_packaged_macos_profile
from jarvis_voice.macos.runtime import MacVoiceRuntime


def main() -> int:
    parser = argparse.ArgumentParser(prog="JARVIS Voice")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / "Library" / "Application Support" / "JARVIS Voice" / "config" / "voice.json",
    )
    args = parser.parse_args()
    config = apply_packaged_macos_profile(load_voice_config(args.config))
    MacVoiceRuntime(config).run()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
