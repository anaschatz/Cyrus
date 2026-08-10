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

## Guided sequence

1. Κράτησε το PTT, πες μία σύντομη ελληνική πρόταση και άφησέ το.
2. Κράτησε το PTT, πες μία σύντομη αγγλική πρόταση και άφησέ το.
3. Πάτησε και άφησε σύντομα το PTT χωρίς να μιλήσεις.
4. Κάνε ακόμη ένα voice turn. Όταν εμφανιστεί η ένδειξη `BARGE-IN NOW` και ακούγεται το TTS, πάτησε και άφησε το PTT χωρίς να μιλήσεις.

Ο runner κλείνει μόνος του μετά το τέταρτο case. `Ctrl-C` δημιουργεί blocked partial evidence αντί για pass.

## Privacy boundary

Το captured audio, το user transcript και τα protocol IDs παραμένουν μόνο στη μνήμη. Το report περιέχει αποκλειστικά language codes, pass/fail flags και monotonic latency deltas. Το fixed fake TTS response στέλνεται στο `/usr/bin/say` μέσω stdin και δεν αποθηκεύεται στο evidence.
