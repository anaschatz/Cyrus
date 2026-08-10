"""Windowed PyInstaller entry point; defaults to the tray command."""

from __future__ import annotations

import multiprocessing
import sys

from jarvis_voice.cli import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    arguments = sys.argv[1:] or ["tray"]
    raise SystemExit(main(arguments))
