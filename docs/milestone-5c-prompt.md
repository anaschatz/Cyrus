# Implementation prompt — Milestone 5C: macOS sealed prerequisite staging

Υλοποίησε το **Milestone 5C — macOS sealed prerequisite staging** του JARVIS Voice Interface.

## Στόχος

Μετέτρεψε τους blockers του M5B σε επαληθεύσιμο, read-only acquisition gate. Το milestone πρέπει να αποδεικνύει αν υπάρχουν ήδη όλα τα inputs για μελλοντική offline δημιουργία isolated Python 3.11 environment, χωρίς να εγκαθιστά, να κατεβάζει ή να εκτελεί το voice runtime.

## Αμετάβλητα όρια

1. Καμία εγκατάσταση Python/package, `pip`, Homebrew mutation, download, model conversion/copy/load, inference, microphone stream, TTS playback ή permission prompt.
2. Ο υποψήφιος Python executable δίνεται ως absolute path και ελέγχεται μόνο με fixed `-I -S` probe. Απαίτησε macOS 14+, arm64, 64-bit CPython 3.11.
3. Το υπάρχον CTranslate2 model επαληθεύεται ξανά με το pinned four-file manifest χωρίς model load.
4. Το Silero artifact πρέπει να υπάρχει ήδη με explicit path/size/SHA-256 στο private config.
5. Προαιρετικό wheelhouse γίνεται δεκτό μόνο μαζί με strict JSON lock manifest για target `python=3.11`, `platform=macos_arm64`, exact top-level pins και exact filename/size/SHA-256 για κάθε wheel. Δεν ακολουθούνται symlinks και κανένα network fallback δεν επιτρέπεται.
6. Ένα hash-locked wheelhouse αποδεικνύει ακεραιότητα του supplied set, όχι ότι η dependency resolution είναι πλήρης. Αυτό αποδεικνύεται αργότερα μόνο από offline install και import/runtime doctor.
7. Το redacted report δεν περιέχει paths, filenames, hashes, usernames, hostname, audio, transcript, prompts ή subprocess output.

## Interface

Πρόσθεσε read-only source entry point:

```text
PYTHONPATH=src python3 scripts/stage_macos_cpu.py
  --config <safe-or-private-config>
  --python <absolute-python-executable>
  --model-dir <absolute-local-ct2-directory>
  [--wheelhouse <absolute-directory>]
  [--wheelhouse-manifest <absolute-json>]
  [--output <redacted-json>]
```

Το report περιλαμβάνει typed checks για Python, wheelhouse, model, VAD, explicit Greek/English voices και non-prompting Input Monitoring preflight. Περιλαμβάνει επίσης `approval_required`, `installation_performed: false`, `runtime_started: false`, `model_loaded: false`, `network_used: false` και ένα content-free `next_action`.

## Completion semantics

Το `ready: true` σημαίνει μόνο «όλα τα inputs για offline install είναι staged». Δεν σημαίνει ότι έγινε install ή native voice smoke. Ακόμη και τότε το `next_action` ζητά explicit approval πριν από environment mutation.

Στο development Mac εκτέλεσε το staging με ό,τι υπάρχει ήδη. Αναμενόμενο blocked αποτέλεσμα δεν είναι failure του harness. Μην δημιουργήσεις dummy wheelhouse ή Silero artifact και μην εμφανίσεις fake pass.

## Verification

- Tests για isolated Python probe, wrong version/platform, absolute-path enforcement.
- Tests για strict wheelhouse schema, exact set, hash/size mismatch και symlink rejection.
- Synthetic fully-ready redacted report χωρίς πραγματικά ML imports.
- Πραγματικό read-only run στο development Mac.
- Όλα τα προηγούμενα tests παραμένουν πράσινα και το protocol version παραμένει `1`.

