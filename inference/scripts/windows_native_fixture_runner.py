#!/usr/bin/env python3
"""PyInstaller entrypoint for the sealed Windows inference fixture runner."""

from jarvis_inference.windows_fixture import main


if __name__ == "__main__":
    raise SystemExit(main())
