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
Η native βαθμονόμηση ανοίγει το πραγματικό CoreAudio input, μετρά πρώτα το
ambient επίπεδο και έπειτα παίζει μία τοπική φράση με την επιλεγμένη macOS
φωνή. Αποθηκεύει μόνο συγκεντρωτικά dBFS στοιχεία και ποτέ PCM:

```sh
python scripts/run_macos_native_collection.py calibrate \
  --config "/absolute/private/voice.json" \
  --audio-path built_in \
  --output "/absolute/private/evidence/acoustic-calibration.json"
```

Η παραγόμενη `calibration.silence_threshold_dbfs` μεταφέρεται αυτούσια στα
`preflight` και `run`. Η βαθμονόμηση αποτυγχάνει κλειστά αν η έξοδος δεν είναι
τουλάχιστον 12 dB πάνω από το ambient επίπεδο στο μικρόφωνο.

Το `run` απαιτεί ρητό `--threshold-calibrated` και γράφει atomic partial evidence
κάθε λεπτό, ώστε ένα crash ή `Ctrl-C` να μην εξαφανίζει αποτυχημένη προσπάθεια.
Το CoreAudio callback κάνει μόνο bounded `put_nowait`· RMS και sustained-silence
ανάλυση γίνονται σε ξεχωριστό thread. Δεν αποθηκεύεται raw audio. Κατά το run,
νεκρό PortAudio stream προκαλεί αυτόματο reopen του CoreAudio device. Wake-sized
heartbeat gap προκαλεί ασφαλές abort του ενεργού turn και επανεκκίνηση τόσο του
CoreAudio stream όσο και του listen-only event tap. Οι πραγματικές επιτυχίες ή
αποτυχίες αυτών των recoveries γράφονται αυτόματα στο native evidence.

## Εγκεκριμένο GreekTTS runtime profile

Το macOS source runtime μπορεί να επιλέξει `macos.tts_backend =
"greektts15_f1"`. Η επιλογή απαιτεί απόλυτα paths για model root, approval
manifest και repository root, καθώς και το ακριβές SHA-256 του approval
manifest. Ο loader επαληθεύει πρώτα το manifest, έπειτα το εγκεκριμένο adapter
hash και τέλος το pinned local model pack. Δεν επιτρέπεται network fallback.

Με `warm_on_start = true` το MLX model και ο SNAC decoder φορτώνονται πριν
αρχίσει το event loop. Το warm load εκτελείται μέσα στο ίδιο μόνιμο inference
worker που παράγει τα audio tokens, επειδή τα Metal streams της MLX είναι
thread-local· timeout ή load failure σταματά κλειστά την εκκίνηση. Τα sentence commits περνούν από τον κοινό epoch-fenced
TTS queue, επομένως conversation και scheduler χρησιμοποιούν την ίδια έξοδο,
ενώ νέο physical `PTT_DOWN` ακυρώνει generation, απορρίπτει pending units και
κάνει άμεσο CoreAudio `abort()`. Η ανθρώπινη έγκριση της φωνής δεν αλλάζει το
native ready gate: acoustic barge-in, realtime factor, 200 turns και soak
παραμένουν υποχρεωτικά.

Το πρώτο πραγματικό approved-checkpoint smoke φόρτωσε warm το μοντέλο σε περίπου
2.9 s και ολοκλήρωσε κανονικά CoreAudio playback με `started` και `completed`
callbacks. Μία μεγαλύτερη, μη εγκεκριμένη διαγνωστική πρόταση απέτυχε κλειστά
στο EOS gate και δεν έπαιξε καθόλου· επομένως sentence robustness και
queue-to-audio latency παραμένουν ανοικτά qualification gates. Το playback
`started` callback εκπέμπεται πριν από το πρώτο blocking PCM write, ώστε οι
native μετρήσεις να περιλαμβάνουν ολόκληρη την πραγματική διάρκεια ήχου.
Η επαναληπτική μέτρηση μετά τη διόρθωση πέρασε με warm load 3.135 s,
queue-to-first-PCM 11.769 s και πραγματικό playback 4.053 s, με ακολουθία
callbacks `started` και `completed` και χωρίς backend error. Η σύνδεση runtime
είναι συνεπώς λειτουργική, αλλά το latency gate παραμένει ανοικτό μέχρι να
αποκωδικοποιούνται και να παίζονται ασφαλή incremental SNAC chunks πριν
ολοκληρωθεί ολόκληρη η πρόταση.

Το πρώτο incremental profile χρησιμοποιεί cumulative SNAC decoding με 36
frames αρχικό buffer, τέσσερα frames look-ahead, τέσσερα frames ανά επόμενο
commit και boundary ramp 120 samples. Κάθε frame αντιστοιχεί σε 2.048 PCM
samples στα 24 kHz. Η επιλογή προέκυψε από πραγματική μέτρηση του εγκεκριμένου
checkpoint: generation RTF 3.587 και prefix drift περίπου 0.0006 ή χαμηλότερο
με το επιλεγμένο look-ahead. Ξεχωριστός bounded playback worker επιτρέπει στη
Metal generation να συνεχίζεται ενώ το blocking CoreAudio stream καταναλώνει
PCM. Το profile παραμένει qualification candidate μέχρι πραγματικό listening,
μηδενικό PortAudio underflow και επιτυχημένο hard barge-in.
