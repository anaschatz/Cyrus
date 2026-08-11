# Native Voice qualification gate

Το production `ready` δεν μπορεί να προκύψει από unit tests, fake backends,
programmatic PTT ή software callback timestamps. Αυτά παραμένουν χρήσιμα smoke
tests, αλλά δεν αποτελούν native acceptance evidence.

## Ελάχιστα κοινά κριτήρια

- τουλάχιστον 200 πραγματικά voice turns με physical PTT, hardware capture και
  hardware playback·
- μηδέν duplicate transcripts, duplicate TTS playbacks και stuck PTT states·
- τουλάχιστον 20 acoustic barge-in δείγματα με ενσύρματο ή built-in local audio·
- από physical `PTT_DOWN` μέχρι το μικρόφωνο μέτρησης να παραμένει κάτω από το
  silence threshold για 50 ms: κάθε δείγμα αυστηρά κάτω από 200 ms·
- τουλάχιστον μία αυτόματη ανάκαμψη από microphone disconnect και μία από
  sleep/resume·
- soak τουλάχιστον 28.800 δευτερολέπτων, χωρίς crash, με τουλάχιστον 97 RSS
  δείγματα·
- μετά το warm-up, RSS slope το πολύ 1 MiB/ώρα και τελικό RSS όχι πάνω από
  32 MiB σε σχέση με την αρχή. Αυτά είναι πρακτικά όρια θορύβου μέτρησης, όχι
  άδεια για συνεχή διαρροή μνήμης.

Η acoustic μέτρηση απαιτεί πραγματικό μικρόφωνο που ακούει την έξοδο. PortAudio
`abort()`, backend-idle callbacks και virtual/electrical loopback δεν μετρούν ως
acoustic silence. Bluetooth και cloud audio paths αποκλείονται από το latency
gate.

## Platform cases

Windows evidence απαιτεί pass για Raw Input, WASAPI, SAPI5, tray,
suspend/resume, lock/unlock, microphone change, missed key-up, packaged EXE,
microphone recovery, sleep recovery και acoustic barge-in. Το runtime profile
πρέπει να είναι signed package με σταθερό SHA-256 fingerprint. Το ξεχωριστό
release gate συνεχίζει να επαληθεύει κάθε PE αρχείο με SignTool.

macOS evidence απαιτεί pass για CoreAudio, Accessibility permission, physical
PTT event tap, τον επιλεγμένο TTS backend, suspend/resume, microphone change,
σταθερό source ή signed-package profile, microphone recovery, sleep recovery
και acoustic barge-in.

## Strict evidence

Δημιούργησε ένα pending template — όχι ψεύτικο pass — σε ήδη υπάρχον private
evidence directory:

```sh
python3 scripts/run_native_voice_qualification.py template \
  --platform macos \
  --output "/absolute/private/evidence/macos-native.json"
```

Αξιολόγηση:

```sh
python3 scripts/run_native_voice_qualification.py evaluate \
  --evidence "/absolute/private/evidence/macos-native.json" \
  --output "/absolute/private/evidence/macos-native-report.json"
```

Το schema απορρίπτει άγνωστα/free-form πεδία. Δεν αποθηκεύει audio, transcript
text, device identifiers ή protocol IDs. Το template ξεκινά με όλα `pending`
και ο evaluator δεν εκτελεί ή προσποιείται hardware tests. Τα τελικά counters
και metrics πρέπει να γραφτούν από τον native telemetry collector της ίδιας
εκτέλεσης. Μέχρι να ολοκληρωθεί αυτός ο collector και να τρέξει στα πραγματικά
μηχανήματα, το αποτέλεσμα παραμένει `qualification.native_blocked`.

## macOS collector

Ο source-profile collector χρησιμοποιεί το υπάρχον απομονωμένο Python 3.11
environment και loopback fixture orchestrator. Το loopback αφορά μόνο τη
νοημοσύνη/απάντηση· κανένα voice-layer component δεν είναι fake. Physical PTT,
CoreAudio capture, πραγματικό ASR και ο επιλεγμένος TTS backend παραμένουν
υποχρεωτικά.

Το preflight δεν ανοίγει μικρόφωνο ή PTT:

```sh
python scripts/run_macos_native_collection.py preflight \
  --config "/absolute/private/voice.json" \
  --audio-path built_in \
  --silence-threshold-dbfs=-40 \
  --output "/absolute/private/evidence/native-collector-preflight.json"
```

Η τιμή `-40` είναι μόνο παράδειγμα και δεν επιτρέπεται σε πραγματικό run μέχρι
να βαθμονομηθεί στο συγκεκριμένο μικρόφωνο, output device, volume και δωμάτιο.
Το `run` απαιτεί ρητό `--threshold-calibrated` και γράφει atomic partial evidence
κάθε λεπτό, ώστε ένα crash ή `Ctrl-C` να μην εξαφανίζει αποτυχημένη προσπάθεια.
Το CoreAudio callback κάνει μόνο bounded `put_nowait`· RMS και sustained-silence
ανάλυση γίνονται σε ξεχωριστό thread. Δεν αποθηκεύεται raw audio.
