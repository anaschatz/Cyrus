# Implementation prompt — Milestone 4: Native Windows release qualification

Υλοποίησε το **Milestone 4 — Native Windows release qualification** του JARVIS Voice Interface στο υπάρχον repository.

## Αμετάβλητα όρια

Το `jarvis.voice/1`, τα PTT/capture/VAD/ASR/transport/TTS/cancellation semantics και τα Milestones 0–3 δεν αλλάζουν. Το M4 δεν προσθέτει νέο voice feature, orchestrator logic, installer, updater ή telemetry. Προσθέτει μόνο αυστηρή, επαναλήψιμη απόδειξη ότι ένα συγκεκριμένο Windows `onedir` artifact μπορεί να χαρακτηριστεί release candidate.

## Κανόνες ακεραιότητας

1. Μην εγκαταστήσεις ή κατεβάσεις dependency, model, driver, SDK, certificate ή voice pack. Μην υπογράψεις artifact, μην κάνεις commit και μην αλλάξεις machine-wide state.
2. Μην παράγεις fabricated pass evidence. Σε macOS/Linux, unsigned artifact, missing SignTool, source-mode doctor, ανεπαρκή samples ή οποιοδήποτε warning, το αποτέλεσμα είναι `ready: false`.
3. Evidence και samples δεν περιέχουν transcript, audio, prompt, exception text, absolute path, username, hostname, device identifier, certificate subject/thumbprint ή SignTool stdout/stderr.
4. Κάθε JSON schema είναι strict: άγνωστα fields, traversal, μη πεπερασμένα/αρνητικά metrics, άγνωστα cases και duplicate/replayed sample IDs απορρίπτονται.
5. Όλες οι writes είναι explicit, atomic και σε user-supplied output path εκτός frozen bundle. Καμία network χρήση ή upload.

## Qualification gates

- **Host:** native Windows 11 x64, CPython 3.11 x64 και feature/API probes των προηγούμενων milestones.
- **Frozen doctor:** παράγεται από το ίδιο packaged EXE μέσω explicit `doctor --output`. Απαιτεί `ready: true`, ίδιο product/protocol/auth version, authenticated loopback transport, `bundle_mode: onedir`, ενεργό DLL hardening, εξωτερικά mutable paths και πράσινο tray/runtime probe.
- **Distribution:** το M3 `distribution-manifest.json` επαληθεύεται αυστηρά και το SHA-256 του καταγράφεται χωρίς path.
- **Authenticode:** εντοπίζεται ήδη εγκατεστημένο SignTool και εκτελείται χωρίς shell ως `verify /pa /all /tw` για κάθε `.exe`, `.dll` και `.pyd`. Μόνο exit code `0` είναι pass· `1`, `2`, timeout ή missing tool είναι blocked. Δεν αποθηκεύεται command output.
- **Smoke samples:** content-free schema με case, outcome και επιτρεπόμενα latency metrics, δεμένο με το SHA-256 του verified distribution manifest. Απαιτεί ελάχιστους επιτυχημένους κύκλους για Greek/English, no-speech, forced release, barge-in first/last unit, keyboard removal, suspend/resume και reconnect deduplication. Οποιοδήποτε failed sample μπλοκάρει release.
- **Latency evidence:** υπολόγισε count, p50 και p95 για PTT_UP→final transcript, PTT_DOWN→SAPI purge, speech commit→SAPI dispatch και backend stream complete→`playback.completed`. Στο M4 δεν επιβάλλεται αυθαίρετο latency SLO· το report καταγράφει τις distributions για approval.

## CLI και artifacts

- `doctor --output <external-json>` για frozen, redacted, atomic doctor evidence.
- `qualification-template --distribution <final-onedir> --output <json>` για κενό strict sample file δεμένο με το συγκεκριμένο artifact.
- `qualification-record --samples <json> --case ... --outcome pass|fail [--metric name=value]` για atomic content-free καταγραφή.
- `qualify-release --config ... --distribution ... --samples ... --frozen-doctor ... --output ...` για το τελικό machine-readable report.
- Native Windows PowerShell runbook/script που δημιουργεί frozen doctor και τελικό report, αλλά δεν κάνει install, build ή signing.

## Acceptance

- Όλα τα tests των M0–M3 παραμένουν πράσινα και το protocol version δεν αλλάζει.
- Fake tests καλύπτουν κάθε positive/negative gate, strict schemas, percentile calculation, SignTool warning/timeout/missing behavior, no-output capture και redaction.
- Native-only tests παραμένουν skipped εκτός Windows/SignTool αντί να δίνουν ψευδές pass.
- Το development-host run παράγει πραγματικό blocked report και αναφέρει ακριβώς γιατί δεν είναι native release evidence.
- Η τελική αναφορά διαχωρίζει source-complete, fake-verified, native-deferred, signature-deferred και sample-deferred αποτελέσματα.
