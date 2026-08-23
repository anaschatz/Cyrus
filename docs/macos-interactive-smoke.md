# macOS interactive CPU smoke

Το M5E χρησιμοποιεί το απομονωμένο environment του M5D και δεν εγκαθιστά τίποτα. Ο runner ξεκινά προσωρινό loopback fake orchestrator, ανοίγει CoreAudio, ενεργοποιεί το passive PTT και εμφανίζει μία οδηγία τη φορά.

## Command

```sh
PYTHONPATH=src \
"/absolute/path/to/venv/bin/python" scripts/run_macos_interactive_smoke.py \
  --config "/absolute/path/to/private/voice.json" \
  --model-dir "/absolute/path/to/local/model/snapshot" \
  --output "/absolute/path/to/private/evidence/macos-interactive-smoke.json"
```

Το default PTT είναι `F9`. Σε πληκτρολόγιο όπου τα function keys ελέγχουν media features μπορεί να χρειάζεται `fn+F9`. Το event tap είναι listen-only: το F9 εξακολουθεί να φτάνει στην foreground εφαρμογή.

Για qualification σε Mac όπου το function row δεν εκπέμπει αξιόπιστα F-key,
χρησιμοποίησε `--ptt-key CONTROL_OPTION`. Το override ισχύει μόνο για το
συγκεκριμένο smoke process και δεν μεταβάλλει το ιδιωτικό production config.
Το `--ptt-key RIGHT_OPTION` παρέχει single-modifier επιλογή όταν το chord
δημιουργεί πολλαπλούς σύντομους capture κύκλους.

## Guided sequence

1. Κράτησε το PTT, πες μία σύντομη ελληνική πρόταση και άφησέ το.
2. Κράτησε το PTT, πες μία σύντομη αγγλική πρόταση και άφησέ το.
3. Πάτησε και άφησε σύντομα το PTT χωρίς να μιλήσεις.
4. Κάνε ακόμη ένα voice turn. Όταν εμφανιστεί η ένδειξη `BARGE-IN NOW` και ακούγεται το TTS, πάτησε και άφησε το PTT χωρίς να μιλήσεις.

Ο runner κλείνει μόνος του μετά το τέταρτο case. `Ctrl-C` δημιουργεί blocked partial evidence αντί για pass.

Το `ready` αυτού του report σημαίνει μόνο ότι πέρασε το σύντομο interactive
smoke. Δεν αποτελεί production-ready gate: δεν μετρά acoustic silence, 200
turns, disconnect/sleep recovery ή 8ωρο soak. Το report το δηλώνει ρητά με
`acoustic_silence_measured: false` και `satisfies_native_ready_gate: false`.

Το production runtime προφορτώνει το επιλεγμένο ASR backend πριν ανοίξει το
PTT, ώστε το πρώτο turn να μην πληρώνει model/sidecar startup. Για κάθε speech
case το report διαχωρίζει πλέον `PTT_UP → transcript final`, `transcript final
→ πρώτο speech commit`, `speech commit → TTS start` και το συνολικό software
`PTT_UP → TTS start`. Έτσι η βελτιστοποίηση μπορεί να στοχεύσει ASR,
orchestrator ή TTS χωρίς αλλαγή μοντέλου ή υποβάθμιση της φωνής.
Το πρώτο σκέλος αναλύεται επιπλέον σε capture close, resampling, VAD, ASR queue,
καθαρό ASR inference και publication του final transcript. Όλα τα markers είναι
content-free monotonic timestamps.

## Privacy boundary

Το captured audio, το user transcript και τα protocol IDs παραμένουν μόνο στη μνήμη. Το report περιέχει αποκλειστικά language codes, pass/fail flags και monotonic latency deltas. Το fixed fake TTS response στέλνεται στο `/usr/bin/say` μέσω stdin και δεν αποθηκεύεται στο evidence.
