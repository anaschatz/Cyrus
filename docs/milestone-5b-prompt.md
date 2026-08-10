# Implementation prompt — Milestone 5B: macOS offline CPU readiness and native smoke

Υλοποίησε το **Milestone 5B — macOS offline CPU readiness and native smoke** του JARVIS Voice Interface στο υπάρχον repository.

## Στόχος

Αξιοποίησε το ήδη επαληθευμένο τοπικό `Systran/faster-whisper-base` CTranslate2 model pack για έναν προσωρινό, ρητό macOS CPU smoke δρόμο. Ο δρόμος αυτός επιτρέπει να ελεγχθεί στο τρέχον Apple Silicon laptop το κοινό pipeline PTT → CoreAudio → SoXR → Silero VAD → ASR → loopback transport → `/usr/bin/say` → cancellation, πριν αποκτηθούν τα χωριστά `whisper.cpp`/GGML artifacts.

Το milestone δεν αντικαθιστά το εγκεκριμένο `whisper.cpp`/Metal profile. Αποδεικνύει μόνο source-run λειτουργικότητα με το ήδη υπάρχον μοντέλο.

## Αμετάβλητα όρια

1. Καμία εγκατάσταση, download, model conversion, model copy, model mutation, network fallback ή permission prompt.
2. Το passive doctor δεν φορτώνει CTranslate2, faster-whisper, ONNX Runtime ή το μοντέλο και δεν ανοίγει audio stream. Επιτρέπονται metadata/version checks, exact manifest hashing, CoreGraphics permission preflight, CoreAudio device query και `/usr/bin/say -v ?`.
3. Το model directory δίνεται ρητά ως απόλυτο path. Η κατασκευή faster-whisper χρησιμοποιεί `device="cpu"` και `local_files_only=True`.
4. Το `faster_whisper_cpu_smoke` είναι explicit opt-in. Το default macOS backend παραμένει `whisper_cpp`; δεν υπάρχει automatic fallback μεταξύ των δύο formats.
5. Το report δεν περιέχει paths, hashes, username, hostname, audio, transcript, prompt, device identifier ή subprocess output.
6. Το protocol παραμένει `jarvis.voice/1`. Το voice layer δεν ερμηνεύει το αίτημα και το `response.speech.done` completion contract δεν αλλάζει.

## Configuration και commands

Πρόσθεσε strict `macos.asr_backend` με ακριβώς δύο τιμές:

- `whisper_cpp` — το υπάρχον Metal-oriented profile και το default.
- `faster_whisper_cpu_smoke` — προσωρινό CPU source-run profile.

Πρόσθεσε:

```text
doctor-macos-cpu
  --config <safe-config>
  --model-dir <absolute-local-ct2-directory>
  [--output <redacted-json>]

run-macos-cpu
  --config <private-config-with-vad-and-voices>
  --model-dir <absolute-local-ct2-directory>
```

Τα commands εφαρμόζουν το opt-in μόνο in-memory. Δεν γράφουν model path στο repository ή σε report.

## Passive readiness semantics

Το doctor ελέγχει:

- macOS 14+, native arm64, 64-bit CPython 3.11,
- `faster-whisper`, `ctranslate2`, `numpy`, `sounddevice`, `soxr`, `onnxruntime`, `websockets`, χωρίς import/model load,
- exact pinned model ID/revision, τέσσερα files, sizes και SHA-256,
- explicit Silero ONNX manifest,
- διαθέσιμο CoreAudio input χωρίς άνοιγμα stream,
- non-prompting passive Input Monitoring preflight,
- explicit εγκατεστημένες `el-GR` και `en-US` system voices,
- loopback-only WebSocket URL.

Η έλλειψη οποιουδήποτε prerequisite επιστρέφει exit `2`, `ready: false` και content-free typed codes. Έγκυρο model pack δεν αρκεί μόνο του για native readiness.

## Native execution semantics

Το `run-macos-cpu` επαναχρησιμοποιεί τον υπάρχοντα macOS event pump και όλα τα κοινά state/protocol invariants. Η μόνη αλλαγή είναι η ρητή κατασκευή `FasterWhisperCpuBackend`. PTT_DOWN συνεχίζει να tombstone-άρει stale output και να ακυρώνει capture/ASR/TTS· ambient speech χωρίς PTT δεν διακόπτει.

Μην ξεκινήσεις το live runtime αυτόματα από το implementation host. Η έναρξη microphone/PTT/TTS απαιτεί πράσινο doctor, explicit private VAD/voice config και συνειδητή interactive εκτέλεση από τον χρήστη.

## Verification

- Unit tests για strict config, absolute model path, explicit backend factory, no implicit fallback και redaction.
- Synthetic ready/blocked doctor tests χωρίς πραγματικό model load ή imports βαριών runtimes.
- Parser tests για τα δύο νέα commands.
- Όλα τα προηγούμενα tests παραμένουν πράσινα και το protocol version παραμένει `1`.
- Εκτέλεσε το πραγματικό passive doctor στο development Mac με το ήδη επαληθευμένο local model directory. Κατέγραψε έντιμα κάθε blocker· μην εμφανίσεις fake native pass.

