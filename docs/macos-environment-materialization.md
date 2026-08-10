# macOS CPU environment materialization

M5D uses the existing `uv` installation to provide user-local CPython 3.11. Runtime packages are first downloaded as wheels, hash-locked and then installed offline into an isolated venv. The repository and the existing Whisper model remain unchanged.

The Silero artifact is extracted from the official `silero-vad` 6.2.1 PyPI wheel without installing the package or its Torch-based dependencies. See the [official Silero release](https://github.com/snakers4/silero-vad/releases/tag/v6.2.1), [PyPI files](https://pypi.org/project/silero-vad/6.2.1/), and [uv Python installation guide](https://docs.astral.sh/uv/guides/install-python/).

The materialization helper exposes four explicit operations:

```text
lock-wheelhouse
extract-silero
write-config
probe-backends
```

It performs no implicit installation or download. Network/package-manager commands remain visible orchestration steps requiring approval. The final interactive PTT run is intentionally outside this milestone.
