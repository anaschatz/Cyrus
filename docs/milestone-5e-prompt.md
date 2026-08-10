# Implementation prompt — Milestone 5E: interactive macOS PTT end-to-end smoke

Υλοποίησε και εκτέλεσε το **Milestone 5E — interactive macOS PTT end-to-end smoke** του JARVIS Voice Interface.

## Στόχος

Απόδειξε σε πραγματικό Apple Silicon Mac ότι το ήδη υλοποιημένο source runtime λειτουργεί end to end με passive push-to-talk, CoreAudio capture, post-capture Silero VAD, το ρητά επιλεγμένο local `faster-whisper` CPU backend, loopback-only protocol transport, sentence-level `/usr/bin/say` TTS και hard barge-in στο επόμενο `PTT_DOWN`.

Το milestone δεν αλλάζει backend, μοντέλο ή protocol version και δεν κάνει νέα εγκατάσταση ή λήψη.

## Native run boundaries

1. Χρησιμοποίησε αποκλειστικά το materialized CPython 3.11 venv, το private config, το ήδη επαληθευμένο `Systran/faster-whisper-base` model pack και το pinned Silero ONNX artifact του M5D.
2. Άνοιξε το πραγματικό microphone μόνο αφού ενημερωθεί ο χρήστης. Επιτρέπεται το macOS microphone permission prompt. Μην αλλάξεις TCC settings μέσω κώδικα ή shell.
3. Χρησιμοποίησε passive CoreGraphics event tap και το configurable function key. Μην καταστείλεις το key και μην προσθέσεις polling ή third-party hotkey dependency.
4. Ο smoke orchestrator πρέπει να δεσμεύεται μόνο σε `127.0.0.1`/loopback, να είναι προσωρινός και να τερματίζεται μαζί με το run.
5. Audio samples και transcript text επιτρέπεται να υπάρχουν μόνο transiently στη μνήμη για VAD/ASR/protocol/TTS. Δεν γράφονται σε file, report ή log.
6. Το `/usr/bin/say` λαμβάνει το fixed fake-response text μέσω stdin. Κανένα user transcript δεν περνά σε argv.
7. Δεν γίνεται network acquisition, model conversion, νέα model λήψη, packaging, staging ή git mutation.

## Guided cases

Εκτέλεσε μία καθοδηγούμενη συνεδρία με την ακόλουθη σειρά:

1. `greek_turn`: ο χρήστης κρατά το PTT, μιλά ελληνικά και το αφήνει. Απαιτούνται VAD speech, final transcript με language `el`, δύο ordered speech commits, `response.speech.done` και `playback.completed` αφού αδειάσει το backend.
2. `english_turn`: ίδια ροή με language `en` και την αγγλική system voice.
3. `no_speech`: σύντομο PTT press/release χωρίς ομιλία. Απαιτείται `input.no_speech` και απουσία νέου transcript/TTS response.
4. `barge_in`: μετά από νέο voice turn ο fake orchestrator παράγει αρκετά μακρύ TTS unit. Μόλις αρχίσει η αναπαραγωγή, ο χρήστης πατά και αφήνει το PTT χωρίς ομιλία. Απαιτούνται `response.cancel`, `playback.interrupted`, epoch cancellation, άδειο TTS backend και κανένα `playback.completed` για το tombstoned response.

Λάθος γλώσσα, runtime/backend error, μη loopback bind, timeout ή μη ολοκληρωμένη διακοπή παράγουν blocked evidence. Δεν επιτρέπεται fake native pass.

## Content-free telemetry and evidence

Κατέγραψε monotonic timestamps μόνο για:

- PTT down/up receipt,
- transcript-final/no-speech creation και language code,
- speech commit/done receipt,
- TTS unit start/complete,
- cancel dispatch,
- playback interrupted/completed,
- backend idle μετά το barge-in.

Το τελικό private JSON report έχει mode `0600` και περιέχει ανά case pass/fail και τις μετρήσεις:

- `ptt_up_to_transcript_final_ms`,
- `speech_commit_to_tts_start_ms`,
- `speech_done_to_playback_completed_ms`,
- `ptt_up_to_no_speech_ms`,
- `ptt_down_to_cancel_dispatch_ms`,
- `ptt_down_to_backend_idle_ms`.

Δεν περιέχει paths, usernames, device identifiers, hashes, event/turn/response IDs, audio, transcript text ή fixed TTS response text.

## Verification and completion

1. Πρόσθεσε deterministic tests για το guided case state machine, metric derivation, failure/timeout semantics, redaction και CLI arguments.
2. Τρέξε όλο το repository test suite πριν από native capture.
3. Τρέξε non-interactive preflight από το materialized venv χωρίς microphone stream.
4. Έπειτα τρέξε τη guided native συνεδρία και γράψε το content-free evidence έξω από το repository.

Το M5E είναι complete μόνο όταν και τα τέσσερα cases περάσουν στο πραγματικό Mac. Αν το macOS αρνηθεί microphone/Input Monitoring, ο χρήστης δεν αλληλεπιδράσει ή κάποιο case αποτύχει, κράτησε το milestone blocked με ακριβή typed code και ασφαλές partial evidence.
