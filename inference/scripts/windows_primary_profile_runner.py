#!/usr/bin/env python3
"""Thin native entrypoint for the sealed Windows primary-profile runner."""

from jarvis_inference.windows_primary_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
