# Pocket TTS English candidate

Pocket TTS is an optional, local-only English backend for the Voice client. It
does not replace `jarvis.voice/1`, interpret response text, or download anything
at runtime. The Orchestrator remains authoritative for every
`response.speech.commit`; the backend only synthesizes and plays the validated
`en-US` speech unit.

## Frozen candidate profile

- Distribution: `pocket-tts==2.1.0`
- Runtime: `torch==2.13.0`, `PyYAML==6.0.3`
- Provider: CPU only
- Sample rate: 24,000 Hz, mono PCM16 playback
- Voice: public `george` embedding
- Temperature: `0.7`
- LSD decode steps: `1`
- Maximum tokens per internal chunk: `50`
- English locale: `en-US`
- macOS fallback: `MacSystemTtsBackend`
- Windows fallback: `SapiTtsBackend`

The runtime requires three absolute, regular, non-symlink artifacts with exact
size and SHA-256. These are the public, non-voice-cloning files used by the
candidate:

| Artifact | Immutable source revision | Bytes | SHA-256 |
|---|---|---:|---|
| `languages/english/model.safetensors` | `kyutai/pocket-tts-without-voice-cloning@d29db7978e464fb90cb3359ee0c69a273b9142cc` | 219029196 | `be9c6b4876d3f30740a8225dfcaa2e43dc4aeb753c15272735bee16bbb4abb0a` |
| `languages/english/tokenizer.model` | same revision | 59339 | `d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6` |
| `languages/english/embeddings/george.safetensors` | `kyutai/pocket-tts-without-voice-cloning@e041936c75475d350b405bc870bcf7c22da4e9e6` | 6243576 | `0c1c6c57c55a98d81254b33728150c7776f40647fe95258d1a6c1a02780b5d02` |

Acquire these artifacts explicitly during provisioning, copy the resolved bytes
out of any Hugging Face symlink cache into an owner-only external model
directory, and verify them before updating Voice configuration. The runtime has
no model ID, URL or cache fallback.

## Installation and configuration

Install the optional locked dependency alongside the platform profile:

```bash
uv sync --extra macos --extra pocket-tts
```

On Windows, combine the normal Windows/audio/transport extras with
`pocket-tts`. Then add the shared artifact profile:

```json
{
  "windows_tts_backend": "pocket_tts",
  "pocket_tts": {
    "model": {
      "path": "/absolute/external/path/model.safetensors",
      "size": 219029196,
      "sha256": "be9c6b4876d3f30740a8225dfcaa2e43dc4aeb753c15272735bee16bbb4abb0a"
    },
    "tokenizer": {
      "path": "/absolute/external/path/tokenizer.model",
      "size": 59339,
      "sha256": "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6"
    },
    "voice_state": {
      "path": "/absolute/external/path/george.safetensors",
      "size": 6243576,
      "sha256": "0c1c6c57c55a98d81254b33728150c7776f40647fe95258d1a6c1a02780b5d02"
    },
    "temperature": 0.7,
    "max_tokens": 50,
    "warm_on_start": true
  },
  "macos": {
    "tts_backend": "pocket_tts"
  }
}
```

Select only the platform field for the machine being provisioned. System voice
mappings remain mandatory because Greek, unsupported locales and a primary
failure route through the existing platform TTS. A failure after Pocket audio
has started never restarts the same sentence through fallback, which prevents
duplicated or mixed speech.

## Runtime and cancellation contract

The model is loaded once on a bounded worker thread. PCM chunks are validated,
converted to signed 16-bit mono and written to a low-latency `sounddevice`
stream as they arrive. `on_started` is emitted only immediately before the
first playable chunk. `on_completed` is emitted only after generation ends and
the output stream drains.

PTT cancellation fences queued units, marks the current generator stale and
aborts the native output stream immediately. Pocket TTS may finish computation
for an already-running internal chunk, but its audio is never accepted after
the epoch tombstone. Text is bounded to 2,048 UTF-8 bytes per authoritative
speech unit.

## Current evidence and remaining gate

Two offline M3/16 GB source smokes with Python 3.11.14 and the exact profile
above produced 2.800–3.040 seconds of audio with:

- 76.8–93.9 ms first generated PCM chunk;
- 4.42–6.84 times realtime synthesis;
- 910.5–910.7 MiB process peak RSS;
- zero network access (`HF_HUB_OFFLINE=1`).

This is candidate evidence, not beta or production qualification. The frozen
technical Mac beta continues to require `system_say`. Promotion requires at
least 100 warm English cases, p95 first audio no greater than 600 ms, p5 speed
at least 2 times realtime, physical PTT-to-silence below 200 ms, exact LLM/ASR
coexistence measurements, restart recovery and an 8-hour soak with zero crash,
OOM, stale speech or monotonic memory growth. Windows requires a separate run on
the RTX 5080 host even though this profile deliberately keeps TTS on CPU.

Voice cloning is not part of this candidate. Kyutai's cloning checkpoint is
gated behind acceptance of its Hugging Face terms; adopting it later would
create a new voice-state/model identity and require separate qualification.
