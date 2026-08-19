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

## Materialize the authenticated single-host configs

Create one external key first. The private parent directory must already exist,
must not be a symlink and the output file must not exist:

```zsh
PYTHONPATH=src python3.11 -m jarvis_voice create-auth-key \
  --output /absolute/private/voice-auth.json
```

Then materialize the Voice config from the safe example and already-qualified
local artifacts:

```zsh
PYTHONPATH=src python3.11 scripts/materialize_macos_cpu.py write-config \
  --base-config config.example.json \
  --model-dir /absolute/local/faster-whisper-base/snapshot \
  --vad-manifest /absolute/private/silero-manifest.json \
  --greek-voice GreekVoiceName \
  --english-voice EnglishVoiceName \
  --voice-auth-key /absolute/private/voice-auth.json \
  --output /absolute/private/voice.local.json
```

Materialize the matching Orchestrator config with the same key path:

```zsh
PYTHONPATH=orchestrator/src:src python3.11 \
  scripts/materialize_orchestrator_config.py \
  --output /absolute/private/orchestrator.local.json \
  --database /absolute/private/state/orchestrator.sqlite3 \
  --voice-auth-key /absolute/private/voice-auth.json
```

All output parent directories and the database parent directory must already
exist. Outputs must not already exist. The helpers validate the
`jarvis.voice-auth-key/1` file and store only its absolute path in config; they
never embed or print its secret. Both configs must continue to reference that
exact same file. The Orchestrator helper obtains each bearer token from 48
operating-system CSPRNG bytes and requires a successful strict production load
both before and after publication. That load checks canonical unpadded
URL-safe-base64 format and a decoded capacity of 32–384 bytes; it does not claim
that randomness can be proven by inspecting a value.

This materialization is for an all-local deployment on this Mac. It does not
authorize a Mac Voice/PC Orchestrator split, and the key must not be copied to
another host. Rotation requires stopping both processes, creating a new key at
a new absolute path, updating both configs and rerunning authenticated
`status`/`e2e` checks plus one native PTT turn after restarting Orchestrator
then Voice. There is no dual-key transition.

The HMAC handshake authenticates the local Voice peers before
`jarvis.voice/1`; it does not encrypt the plain loopback WebSocket, provide TLS,
sandbox processes or protect against same-user compromise.
