# macOS automated fixture smoke

Το M5E-A εκτελεί την πραγματική local ASR/VAD/protocol/system-TTS αλυσίδα χωρίς microphone ή physical hotkey. Τα fixed Greek/English fixtures δημιουργούνται προσωρινά με τις εγκατεστημένες voices, μετατρέπονται σε PCM16/16 kHz και διαγράφονται αυτόματα.

```sh
PYTHONPATH=src \
"/absolute/path/to/venv/bin/python" scripts/run_macos_automated_smoke.py \
  --config "/absolute/path/to/private/voice.json" \
  --model-dir "/absolute/path/to/local/model/snapshot" \
  --output "/absolute/path/to/private/evidence/macos-automated-smoke.json"
```

Κατά την εκτέλεση ακούγονται μόνο fixed test responses από το `/usr/bin/say`. Δεν αποθηκεύεται user content και δεν χρησιμοποιείται δίκτυο. Ένα ready αποτέλεσμα πιστοποιεί την deterministic native pipeline, όχι CoreAudio capture ή physical PTT.

## Pinned Silero contract

Το πρώτο native fixture run αποκάλυψε ότι το exact `silero-vad 6.2.1` ONNX wrapper απαιτεί rolling context 64 samples πριν από κάθε frame 512 samples στα 16 kHz. Ο κοινός adapter εφαρμόζει πλέον αυτό το contract και υπάρχει regression test που ελέγχει διαδοχικά model inputs 576 samples. Η διόρθωση αφορά και τα Windows/macOS runtime paths που μοιράζονται τον ίδιο VAD adapter.
