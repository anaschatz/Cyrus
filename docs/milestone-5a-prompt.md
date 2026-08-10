# Implementation prompt — Milestone 5A: Local artifact compatibility audit

Υλοποίησε το **Milestone 5A — Local artifact compatibility audit** του JARVIS Voice Interface στο υπάρχον repository.

## Στόχος

Εντόπισε και ταξινόμησε τα ήδη υπάρχοντα τοπικά speech artifacts πριν από οποιαδήποτε εγκατάσταση, μετατροπή ή λήψη. Το audit πρέπει να απαντά με ασφάλεια:

- υπάρχει πλήρες faster-whisper/CTranslate2 model pack για το Windows runtime;
- υπάρχει whisper.cpp-compatible model και `whisper-server` για το macOS runtime;
- υπάρχει OpenAI/PyTorch Whisper checkpoint που θα απαιτούσε μελλοντική μετατροπή;
- υπάρχει Silero ONNX VAD artifact;
- ποια ακριβή paths/sizes/hashes μπορούν αργότερα να μπουν σε ιδιωτικό config;

## Αμετάβλητα όρια

1. Καμία εγκατάσταση, download, conversion, model load, inference, import βαριού ML runtime, permission prompt ή network χρήση.
2. Καμία αλλαγή, rename, copy, chmod ή metadata write στα candidate artifacts. Το scan είναι read-only.
3. Μην σαρώσεις αυθαίρετα όλο το filesystem. Απαίτησε ένα ή περισσότερα explicit absolute `--root`, απέρριψε `/`, μην ακολουθείς directory symlinks και βάλε όρια σε depth, directories και candidates.
4. Παράλειψε γνωστά privacy-sensitive/άσχετα trees όπως Mail, Messages, Safari, Keychains, Photos libraries, `.git`, `node_modules` και trash.
5. Το redacted report δεν περιέχει path, filename, username, hostname, hash ή file content. Τα paths επιτρέπονται μόνο σε ξεχωριστό explicit private manifest, με atomic write και mode `0600`.
6. Μην ανοίξεις arbitrary αρχεία. Η default discovery χρησιμοποιεί μόνο directory entries, file metadata και exact filename/layout rules. Content διαβάζεται μόνο με explicit `--hash-candidates`, αποκλειστικά για SHA-256 των ήδη ταξινομημένων candidates.

## Classification rules

- **`faster_whisper_ct2`**: directory που περιέχει και τα τέσσερα runtime artifacts `config.json`, `model.bin`, `tokenizer.json`, `vocabulary.txt`. Επιτρέπονται Hugging Face file symlinks, αλλά όχι directory traversal μέσω symlink.
- **`whisper_cpp_model`**: regular file με approved Whisper-oriented naming (`ggml-*.bin`, `whisper*.gguf`, `ggml*.gguf`). Πρόκειται για compatibility candidate, όχι απόδειξη ότι φορτώνει.
- **`openai_whisper_checkpoint`**: exact Whisper checkpoint naming όπως `base.pt`, `small.pt`, `medium.pt`, `large-v*.pt`, `turbo.pt`. Σημείωσε `conversion_required`, ποτέ αυτόματη μετατροπή.
- **`silero_vad_onnx`**: `silero*.onnx`.
- **`whisper_cpp_server`**: executable regular file με όνομα `whisper-server`.

Μην ταξινομήσεις γενικά `model.bin`, `.pt`, `.onnx` ή `.gguf` αρχεία που δεν ταιριάζουν στους παραπάνω περιορισμούς.

## Outputs

Πρόσθεσε CLI:

```text
artifact-audit
  --root <absolute-dir> [--root ...]
  --output <redacted-json>
  --private-manifest <private-json>
  [--max-depth N]
  [--max-directories N]
  [--max-candidates N]
  [--hash-candidates]
```

Το private manifest περιέχει schema version, explicit roots και candidates με deterministic ID, kind, absolute path, size, compatibility, symlink status και προαιρετικό SHA-256. Δηλώνει καθαρά `path_disclosure: true`.

Το redacted report περιέχει μόνο counts ανά kind/compatibility, scan counters, truncation/unreadable status, αν γράφτηκε private manifest και content-free next-action code. Δηλώνει `network_used: false` και αν έγινε hashing.

## Failure semantics

- Missing/non-directory/relative root, `/`, ίδιο output/private path, invalid limits ή output symlink: typed failure, exit `2`.
- Directory/candidate limit: deterministic `completed: false`, `truncated: true`, exit `2`. Τα partial candidates παραμένουν εμφανώς μη authoritative.
- Permission/stat/hash error: content-free counter/code, ποτέ exception text ή path στο redacted report.
- Κανένα candidate δεν είναι έγκυρο audit completion, όχι runtime pass: `completed: true` με next action `artifact.whisper_not_found`.

## Verification

- Fake filesystem tests για όλες τις κατηγορίες, exact CT2 layout, false-positive rejection, symlink handling, bounds, deduplication, optional hashing, atomic/private permissions και redaction.
- CLI test με πραγματικό temporary tree, χωρίς network/import/model load.
- Όλα τα M0–M5 tests παραμένουν πράσινα και το protocol version παραμένει `1`.
- Εκτέλεσε το audit στα πραγματικά, explicit local roots που υπάρχουν στο development laptop. Μην υποθέσεις ότι το υπάρχον μοντέλο είναι συμβατό με δύο runtimes· ανέφερε μόνο ό,τι αποδεικνύει το layout.
