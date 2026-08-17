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

Το πρώτο native incremental smoke βρήκε PortAudio underflow πριν από το τελικό
chunk. Η content-free χρονοσειρά έδειξε κενό 291 ms: το synchronous CPU decode
σταματούσε προσωρινά τη Metal generation. Το pipeline χωρίστηκε συνεπώς σε
τρεις bounded, cancellation-aware βαθμίδες — Metal token generation, CPU SNAC
decoding και CoreAudio playback. Ενδιάμεσο decode μπορεί να παραλειφθεί όταν η
bounded queue είναι γεμάτη, επειδή το επόμενο cumulative snapshot περιλαμβάνει
όλα τα ακόμη μη εκπεμφθέντα stable samples· το τελικό decode και completion δεν
παραλείπονται ποτέ.

Το δεύτερο πραγματικό incremental CoreAudio smoke πέρασε μετά τον διαχωρισμό
των τριών βαθμίδων: warm load 3.217 s, queue-to-first-PCM 11.441 s και συνεχές
playback 4.037 s, με callbacks `started` και `completed`, χωρίς PortAudio
underflow ή backend error. Το queue-to-audio βελτιώθηκε κατά περίπου 328 ms σε
σχέση με το προηγούμενο full-sentence runtime smoke. Το automated pipeline gate
πέρασε, αλλά το incremental quality gate παραμένει ανοικτό μέχρι ανθρώπινη
ακρόαση για boundary clicks, ρομποτικό timbre ή pacing regression και μέχρι
πραγματικό acoustic barge-in.

Human listening approved the second incremental smoke as clean and continuous,
without a robotic boundary artifact. The approval manifest therefore records
the incremental streaming gate as passed. Production promotion remains false;
hard interruption, realtime factor and native scheduler qualification remain
open.

### Native F1 hard-cancellation smoke

The first native cancellation run exposed a playback lifecycle defect: the
CoreAudio stream stopped in 106.7 ms, but a single large blocking PCM write kept
the playback worker alive until its 10-second timeout and produced
`tts.greektts15.runtime_failed`. PCM delivery was therefore bounded to 480-sample
(20 ms) CoreAudio writes before repeating the same test.

The corrected native Metal/CoreAudio run cancelled 505.1 ms after the first PCM
write. `cancel()` returned in 106.6 ms and the complete backend reached idle in
112.9 ms. The callback sequence contained only `started`; there was no late
`completed`, stale playback or runtime error. This passes the software
hard-cancellation check. It does not yet set the acoustic barge-in gate: the
ready threshold must still be measured from PTT activation to captured speaker
silence with a physical microphone and wired/local audio path.

### Built-in F1 acoustic calibration

The selected digest-bound F1 backend was calibrated through the actual MacBook
Air microphone and speakers without retaining PCM. The accepted run measured
ambient p95 at -56.303 dBFS, audible playback p80 at -39.856 dBFS and 16.448 dB
of separation. The resulting sustained-silence threshold is -53.303 dBFS with
a 3 dB audible margin.

An earlier run exposed that playback p20 was dominated by natural between-word
pauses. Calibration now uses audible p80 speech frames and separately enforces
ambient/playback separation; regression coverage includes a pause-heavy sample.

The first 120-second physical collection attempt produced zero PTT turns with
the MacBook F9 configuration even though the passive Input Monitoring doctor
reported ready. The likely cause is the top-row media-key mapping. This run is
blocked evidence, not an acoustic sample; the next source qualification attempt
uses the already-supported Control+Option chord override.

The focused native harness then removed ASR and orchestrator timing from the
measurement and waited only for one physical Control+Option press during active
F1 playback. The first valid built-in-audio sample measured 57.861 ms from the
physical event-tap receipt to microphone-observed sustained silence; the
backend `cancel()` call returned in 104.432 ms. Both are below the 200 ms ready
limit. No PCM was retained. This is one valid acoustic sample, so the strict
gate remains open until 20 real samples pass with no invalid observation.

For the 16 GB M3 qualification machine, `scripts/run_macos_f1_acoustic_qualification.py`
is the required F1 acoustic entry point. Its `batch` command hard-limits each
process to two samples, writes atomic content-free evidence, and exits so MLX,
SNAC and CoreAudio are fully unloaded before the next batch. Run `calibrate`
after an environment or output-volume change, then use its exact
`silence_threshold_dbfs` for each two-sample batch. Twenty-sample evidence must
come from ten successful bounded processes; a failed batch resets the gate.

The approved F1 runtime keeps MLX/SNAC inference in the client process but owns
PortAudio/CoreAudio playback in a disposable `spawn` helper. PCM crosses an
anonymous process pipe and is never persisted. Barge-in terminates and joins
the helper instead of invoking `RawOutputStream.abort()` concurrently with a
blocking native write. A fresh helper is created for the next speech unit, so
late audio and a native playback fault cannot escape the response epoch or
crash the main voice client.

The final built-in-audio gate passed on 2026-08-17 with ten independent
two-sample processes and the exact isolated F1 runtime. All 20/20 physical
Control+Option interruptions were valid and below 200 ms: mean 87.931 ms,
nearest-rank p95 161.573 ms, and maximum 172.462 ms. The maximum synchronous
cancel call was 5.73 ms. There were no duplicate completions, stale playback or
crashes, and the user confirmed clean audio quality. The content-free aggregate
and source-report digests are recorded in
`docs/evidence/macos-f1-acoustic-qualification-20260817.json`. This closes the
F1 `interrupt_during_generation` runtime gate; production promotion remains
blocked by the real-time-factor and native scheduler gates.
