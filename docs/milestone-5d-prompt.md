# Implementation prompt — Milestone 5D: approved macOS CPU environment materialization

Υλοποίησε και εκτέλεσε το **Milestone 5D — approved macOS CPU environment materialization** του JARVIS Voice Interface.

## Στόχος

Με ρητή έγκριση του χρήστη, δημιούργησε user-local isolated CPython 3.11 environment για το M5B CPU smoke, απέκτησε και κλείδωσε τα εγκεκριμένα runtime wheels, υλοποίησε το Silero ONNX artifact και απέδειξε ότι ASR/VAD/audio dependencies φορτώνουν τοπικά χωρίς να ξεκινήσεις ακόμη interactive PTT/audio/TTS runtime.

## Acquisition boundaries

1. Χρησιμοποίησε το ήδη εγκατεστημένο `uv`, χωρίς update και χωρίς Homebrew mutation, για μία user-local managed CPython 3.11 έκδοση. Κατέγραψε την ακριβή patch version μετά την εγκατάσταση.
2. Έλεγξε διαθέσιμο χώρο πριν από κάθε acquisition. Μην ξεκινήσεις αν υπάρχουν λιγότερα από 2 GiB. Μη διαγράψεις caches ή υπάρχοντα environments χωρίς ξεχωριστή έγκριση.
3. Όλα τα mutable runtime artifacts μένουν κάτω από ιδιωτικό per-user development directory έξω από το repository. Μην αλλάξεις global `PATH`, shell profile ή `.python-version`.
4. Κατέβασε με `pip download --only-binary=:all: --no-cache-dir` τα exact direct requirements του `requirements/macos-cpu.in` και όλες τις transitives σε wheelhouse. Μην επιτρέψεις source builds.
5. Δημιούργησε strict wheelhouse lock με exact file set, distribution metadata, size και SHA-256. Η τελική εγκατάσταση στο venv γίνεται αποκλειστικά με `pip install --no-index --find-links ... --no-cache-dir`.
6. Για Silero χρησιμοποίησε μόνο το επίσημο PyPI wheel `silero_vad-6.2.1-py3-none-any.whl` με το δημοσιευμένο SHA-256 `09de93c4d874bb19c53e62a47dd38be5f163cedad2b5599583231f2a84ef79cb`. Κατέβασέ το με `--no-deps`, μην εγκαταστήσεις το package/Torch και εξήγαγε μόνο `silero_vad/data/silero_vad.onnx` με bounds και νέο private manifest.
7. Χρησιμοποίησε τις ήδη εγκατεστημένες voices `Melina` για `el-GR` και `Samantha` για `en-US`. Μην κατεβάσεις φωνές.
8. Μην ζητήσεις ή αλλάξεις TCC permissions. Μην ανοίξεις microphone stream και μην παίξεις TTS σε αυτό το milestone.

## Exact requirements

- `faster-whisper==1.2.1`
- `sounddevice==0.5.3`
- `soxr==1.1.0`
- `onnxruntime==1.27.0`
- `websockets==15.0.1`

Η επιλογή παραμένει `device="cpu"`, `local_files_only=True` και το ήδη επαληθευμένο `Systran/faster-whisper-base` model pack. Κανένα νέο ASR model download.

## Verification

1. `pip check` στο isolated venv.
2. Offline staging με Python executable, wheelhouse lock, model, VAD και private voice config.
3. Redacted doctor από το Python 3.11 environment.
4. Non-interactive backend probe: φόρτωσε faster-whisper model, φόρτωσε ONNX Runtime session, εκτέλεσε μόνο ένα synthetic zero frame και κάνε CoreAudio device query χωρίς stream.
5. Reports δηλώνουν αν έγινε network acquisition, αλλά δεν περιέχουν paths, hashes, wheel filenames, device identifiers ή model/audio content.
6. Όλα τα repository tests παραμένουν πράσινα. Το protocol παραμένει `1`.

## Completion

Το milestone είναι complete όταν το venv είναι λειτουργικό και το backend load probe περνά. Input Monitoring και το interactive Greek/English PTT smoke παραμένουν το επόμενο manual/native gate. Αν dependency resolution, Silero contract, disk space ή native wheel availability αποτύχει, σταμάτα με ακριβές typed evidence· μην κάνεις unpinned fallback ή source build.
