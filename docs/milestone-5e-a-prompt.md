# Implementation prompt — Milestone 5E-A: deterministic native fixture qualification

Υλοποίησε και εκτέλεσε το **Milestone 5E-A — deterministic native fixture qualification** του JARVIS Voice Interface στο πραγματικό Apple Silicon Mac.

## Γιατί υπάρχει

Το interactive M5E άνοιξε επιτυχώς CoreAudio και passive PTT, αλλά έληξε χωρίς φυσικό `PTT_DOWN/UP`. Το M5E-A δεν αντικαθιστά ούτε «περνά» αυτό το manual gate. Απομονώνει και πιστοποιεί αυτοματοποιημένα όλη την υπόλοιπη native αλυσίδα, ώστε να ξεχωρίσει καθαρά ένα interaction/TCC ζήτημα από ASR, VAD, protocol, TTS ή cancellation failure.

## Στόχος

Με deterministic, fixed-content speech fixtures που δημιουργούνται τοπικά από τις ήδη εγκατεστημένες macOS voices:

1. πέρασε ελληνικό και αγγλικό PCM audio από το πραγματικό pinned Silero ONNX και το υπάρχον local `faster-whisper-base` CPU backend,
2. επαλήθευσε language metadata `el` και `en` χωρίς να αποθηκεύσεις transcript text,
3. πέρασε strict ordered sentence commits και `response.speech.done` από το πραγματικό `VoiceCore`,
4. παίξε fixed response units με το πραγματικό `/usr/bin/say` backend και απαίτησε `playback.completed` μόνο όταν ολοκληρωθούν όλα τα units και το backend γίνει idle,
5. επαλήθευσε VAD no-speech με synthetic zero PCM,
6. εκτέλεσε deterministic hard barge-in ενώ μιλά το `/usr/bin/say` και απαίτησε cancel dispatch, `playback.interrupted`, backend idle και απόρριψη stale completion.

## Boundaries

- Καμία εγκατάσταση, λήψη, model conversion ή network access.
- Μην ανοίξεις microphone και μην ενεργοποιήσεις physical/global PTT.
- Χρησιμοποίησε αποκλειστικά το materialized CPython 3.11 venv, private config, local model και pinned VAD artifact του M5D.
- Δημιούργησε τα fixtures με `/usr/bin/say` μέσω stdin και μετέτρεψέ τα σε mono PCM16/16 kHz μέσω `/usr/bin/afconvert`, χωρίς shell.
- Τα fixture files ζουν μόνο σε private temporary directory, έχουν bounded size/duration και διαγράφονται σε κάθε success/failure path.
- Το output TTS επιτρέπεται να ακουστεί, αλλά περιέχει μόνο fixed test content.
- Το evidence δεν περιέχει audio, transcript/TTS text, paths, usernames, hashes, filenames, voice/device identifiers ή protocol IDs.

## Required cases

- `greek_fixture`: VAD speech, ASR language `el`, two ordered commits, speech done, TTS start/complete και playback completed.
- `english_fixture`: τα αντίστοιχα με language `en`.
- `silence_fixture`: `input.no_speech`, χωρίς ASR/TTS response.
- `programmatic_barge_in`: ενεργό long TTS unit, νέο PTT epoch, response cancel, playback interrupted, backend idle και κανένα playback completed για το tombstoned response.

## Evidence

Γράψε atomic private JSON mode `0600` με case status και μόνο content-free timings:

- `fixture_to_transcript_final_ms`,
- `speech_commit_to_tts_start_ms`,
- `speech_done_to_playback_completed_ms`,
- `silence_vad_ms`,
- `ptt_down_to_cancel_dispatch_ms`,
- `ptt_down_to_backend_idle_ms`.

Το report δηλώνει υποχρεωτικά:

- `fixture_injected: true`,
- `microphone_opened: false`,
- `physical_ptt_exercised: false`,
- `satisfies_interactive_m5e: false`,
- `network_used: false`.

## Completion

Το M5E-A είναι complete όταν όλα τα deterministic cases περάσουν στο πραγματικό Mac. Το interactive M5E παραμένει blocked μέχρι να ολοκληρωθούν τα τέσσερα physical PTT cases από άνθρωπο. Απαγορεύεται να συγχωνευτούν ή να παρουσιαστούν τα δύο gates ως ισοδύναμα.
