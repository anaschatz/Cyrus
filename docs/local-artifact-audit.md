# Milestone 5A — Local artifact compatibility audit

This command discovers already-local speech artifacts. It does not install, download, convert, load or execute a model and it never contacts a registry or model hub.

## Metadata-only discovery

Use only explicit absolute roots. Keep the redacted report separate from the path-bearing private manifest:

```zsh
mkdir -p /private/tmp/jarvis-artifact-evidence

PYTHONPATH=src python3 -m jarvis_voice artifact-audit \
  --root /explicit/local/model/cache \
  --root /explicit/local/project/models \
  --output /private/tmp/jarvis-artifact-evidence/summary.json \
  --private-manifest /private/tmp/jarvis-artifact-evidence/private.json
```

The default pass reads directory entries and file metadata only. It rejects relative roots and `/`, does not follow directory symlinks, skips privacy-sensitive trees and stops at configured directory/candidate bounds.

`summary.json` is content-free and contains only counts, compatibility categories, scan counters and a next-action code. `private.json` is written atomically with mode `0600` and contains absolute paths. Treat it as private local configuration material.

## Explicit SHA-256 pass

After reviewing the private candidate list, hashing can be requested explicitly:

```zsh
PYTHONPATH=src python3 -m jarvis_voice artifact-audit \
  --root /explicit/local/model/cache \
  --output /private/tmp/jarvis-artifact-evidence/summary-hashed.json \
  --private-manifest /private/tmp/jarvis-artifact-evidence/private-hashed.json \
  --hash-candidates
```

Only files already classified by the exact rules are opened. Dataless File Provider placeholders and file symlinks resolving outside every explicit root are not read. A failed or partial hash leaves `completed: false`.

## Interpretation

- `faster_whisper_ct2` proves only that the expected runtime artifacts for a supported profile are present (four for base, five for large-v3-turbo). Exact hashes must still match the pinned project manifest before loading.
- `whisper_cpp_model` is a filename/layout compatibility candidate, not a successful native load probe.
- `openai_whisper_checkpoint` is never converted automatically.
- `whisper_cpp_server` is only an existing executable candidate. M5 still requires the approved revision and artifact manifest.
- No candidates is a successfully completed audit with `artifact.whisper_not_found`, not a runtime pass.

An audit with unreadable directories or bounds reached is deliberately incomplete. Narrow the roots and rerun instead of treating partial discovery as authoritative.

## Development-laptop result

The M5A run on the explicit Hugging Face, Downloads and Documents roots found one complete multilingual `Systran/faster-whisper-base` CTranslate2 pack and no macOS whisper.cpp model, `whisper-server` or Silero ONNX artifact. An explicit four-file SHA-256 pass matched the existing pinned `BASE_MODEL_MANIFEST` exactly. Therefore the already-local model is ready as a Windows faster-whisper candidate; it is not a whisper.cpp model and is not modified or converted by this milestone.
