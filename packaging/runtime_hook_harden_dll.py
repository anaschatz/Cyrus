"""Runs before application imports in a frozen Windows process."""

from jarvis_voice.app_paths import frozen_library_directory
from jarvis_voice.windows.hardening import apply_frozen_dll_policy

library_directory = frozen_library_directory()
if library_directory is not None:
    apply_frozen_dll_policy(library_directory)
