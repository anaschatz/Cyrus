# Implementation prompt — Milestone 5: macOS Apple Silicon parity

Υλοποίησε το **Milestone 5 — macOS Apple Silicon parity** του JARVIS Voice Interface στο υπάρχον repository.

## Στόχος

Πρόσθεσε native source-run voice path για `macOS >= 14`, Apple Silicon (`arm64`) και CPython 3.11, χωρίς να αλλάξεις το `jarvis.voice/1` ή τα PTT/capture/VAD/transport/TTS/cancellation semantics των προηγούμενων milestones. Το macOS layer παραμένει απλή διεπαφή: δεν ερμηνεύει αιτήματα και δεν προσθέτει orchestrator logic.

## Αμετάβλητα όρια

1. Μην εγκαταστήσεις ή κατεβάσεις dependency, model, binary, voice ή SDK. Χρησιμοποίησε μόνο explicit τοπικά paths και manifest validation.
2. Μην πειράξεις το ανεξάρτητο Windows runtime. Τα Windows `faster-whisper`/CUDA/SAPI/Raw Input paths παραμένουν λειτουργικά και import-safe.
3. Μην προσθέσεις wake word, hands-free VAD interruption, AEC, cloud ASR/TTS, telemetry, installer ή updater.
4. Το v1 barge-in παραμένει στο `PTT_DOWN`: πρώτα tombstone/cancel/purge και μετά capture. Το passive PTT δεν καταστέλλει το function key.
5. Καμία transcript/audio/prompt πληροφορία σε logs, doctor output, subprocess argv ή προσωρινά αρχεία. Καμία network πρόσβαση εκτός του ήδη επιτρεπόμενου loopback orchestrator και του owned loopback ASR sidecar.

## Platform adapters

### Capture

- `python-sounddevice` `RawInputStream` πάνω σε CoreAudio, mono `float32`, native device sample rate, `blocksize=0`, `latency="low"`.
- `CoreAudioSettings(change_device_parameters=False, fail_if_conversion_required=False, conversion_quality="max")`.
- Το callback μόνο αντιγράφει σε bounded non-blocking buffer. Το υπάρχον SoXR worker path κάνει normalize σε 16 kHz μετά το PTT_UP.

### Passive PTT

- Native CoreGraphics listen-only event tap, χωρίς `pynput` και χωρίς suppression.
- Υποστήριξε F1–F20, αγνόησε autorepeat, timestamp με `perf_counter` στον callback και δημοσίευσε μέσω της υπάρχουσας bounded fail-safe `PttEventPublisher`.
- Event-tap disable/timeout και shutdown ενώ το key είναι held προκαλούν forced release. Μην εμφανίζεις αυτόματα TCC prompt· το doctor κάνει preflight και δίνει content-free permission code.

### Local ASR

- Owned, long-lived, unprivileged `whisper-server` sidecar από pinned local executable/model artifacts με size και SHA-256.
- Bind αποκλειστικά σε `127.0.0.1`, τυχαία διαθέσιμη υψηλή port και νέο μη αναφερόμενο request-path token ανά child. Metal/flash attention on, no-context, task transcribe, language `auto` ή explicit ISO code. Core ML off στο v1.
- Στείλε ήδη trimmed mono 16-kHz PCM ως in-memory WAV multipart στο ιδιωτικό `/inference` path. Μην ενεργοποιήσεις ffmpeg conversion, upload-to-disk ή initial prompt.
- Βάλε όρια σε audio/request/response size, startup/request timeout και strict response parsing. Σε close/timeout/crash τερμάτισε το owned child. Μην αποθηκεύεις stdout/stderr ή paths σε errors.
- Η έλλειψη binary/model δεν προκαλεί download ή fallback σε Hugging Face. Παραμένει typed blocker.

### Streaming TTS και cancellation

- Αρχικό macOS backend είναι το `/usr/bin/say`, ένα process ανά committed speech unit, σε ordered worker queue.
- Το text περνά μόνο από stdin (`-f -`), ποτέ από argv ή αρχείο. Locale-to-voice mapping είναι explicit.
- `cancel(new_epoch, reason)` απορρίπτει pending stale units και κάνει άμεσο terminate/kill του ενεργού process. Late completion μετά το tombstone αγνοείται.
- Το `playback.completed` εξακολουθεί να απαιτεί `response.speech.done`, ολοκλήρωση όλων των units, άδεια queue/backend και μη cancelled response.

## Configuration και diagnostics

- Πρόσθεσε strict `macos` config section για executable/model manifests, language, threads, timeouts, response cap και voice mapping.
- Πρόσθεσε `doctor-macos --config ... [--output ...]` και `run-macos --config ...` χωρίς να αλλάξεις το υπάρχον `doctor`/`run` των Windows.
- Το macOS doctor είναι offline/redacted και ελέγχει Darwin arm64, macOS 14+, Python 3.11, local artifacts, CoreGraphics listen permission, CoreAudio dependency/device, SoXR/ONNX/WebSocket dependencies και τις configured system voices.
- Source mode ή λάθος Python/permissions/missing artifacts δίνει `ready: false`, όχι fabricated pass.

## Verification

- Fake tests για CoreAudio arguments/callback boundary, key mapping/autorepeat/forced release, sidecar argv και loopback-only HTTP multipart, response limits/parsing, TTS stdin/privacy/order/cancel races, config validation και redacted doctor.
- Native-only permission/audio/sidecar/model/TTS tests είναι skipped όταν λείπει explicit local prerequisite.
- Όλα τα M0–M4 tests παραμένουν πράσινα και το protocol version παραμένει `1`.
- Στο development Mac εκτέλεσε πραγματικό `doctor-macos`. Αν λείπουν Python 3.11, dependencies ή explicit artifacts, η αναμενόμενη έντιμη έκβαση είναι blocked με content-free codes.

## Packaging boundary

Signing, notarization, `.app`, global TCC identity και model distribution παραμένουν επόμενο milestone. Το M5 παραδίδει source-complete/fake-verified adapters και native diagnostic evidence, όχι release artifact.
