# macOS sealed prerequisite staging

Run this before approving any Python or package installation:

```zsh
PYTHONPATH=src python3 scripts/stage_macos_cpu.py \
  --config config.example.json \
  --python /absolute/path/to/python3.11 \
  --model-dir /absolute/path/to/the/local/faster-whisper-base/snapshot \
  --output /absolute/private/macos-staging.json
```

Optionally add `--wheelhouse` and `--wheelhouse-manifest` together. The manifest schema is strict:

```json
{
  "schema_version": 1,
  "target": {"python": "3.11", "platform": "macos_arm64"},
  "top_level": {
    "faster-whisper": "1.2.1",
    "sounddevice": "0.5.3",
    "soxr": "1.1.0",
    "onnxruntime": "1.27.0",
    "websockets": "15.0.1"
  },
  "wheels": [
    {"filename": "resolved-wheel.whl", "size": 123, "sha256": "lowercase-sha256"}
  ]
}
```

List every wheel present in the directory and no others. This gate hashes the supplied artifacts but never invokes pip. A later approved installer must still use `--no-index`, an isolated venv and a post-install import doctor.
