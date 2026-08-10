# Implementation prompt — Milestone 3: Windows packaging, tray and hardening

Υλοποίησε το **Milestone 3 — Windows packaging, tray και hardening** του JARVIS Voice Interface στο υπάρχον repository.

## Αμετάβλητο όριο ευθύνης

Το voice interface αναλαμβάνει αποκλειστικά PTT, capture, VAD, transcription, loopback transport, TTS και cancellation. Δεν ερμηνεύει αιτήματα, δεν επιλέγει εργαλεία και δεν περιέχει agent/orchestrator logic. Το `jarvis.voice/1`, τα epochs/tombstones, το `response.speech.done` και τα failure semantics των Milestones 0–2 δεν αλλάζουν.

## Στόχος

Παράδωσε ένα ελέγξιμο Windows 11 x64 `onedir` packaging path και ένα ελαφρύ native Win32 tray shell γύρω από το ήδη υπάρχον runtime. Πρόσθεσε lifecycle, per-session single-instance protection, ασφαλή application directories, περιορισμό DLL search, redacted local event log και deterministic distribution manifest.

## Υποχρεωτικές αποφάσεις

1. **PyInstaller `onedir`, windowed, asInvoker.** Το build γίνεται μόνο σε native Windows 11 x64 με CPython 3.11. Δεν υπάρχει cross-build από macOS, `onefile`, elevation ή UPX.
2. **Καμία εγκατάσταση ή λήψη.** Μην εγκαταστήσεις dependencies, models, drivers, CUDA/cuDNN, voices ή packaging tools. Μην κάνεις network fallback, signing, commit ή machine-wide αλλαγή.
3. **Mutable data εκτός bundle.** Config, logs και state βρίσκονται κάτω από `%LOCALAPPDATA%\JARVIS Voice`. Το default models directory είναι επίσης εκεί, αλλά ένα ήδη υπάρχον explicit local model/VAD path επιτρέπεται αλλού μόνο εφόσον βρίσκεται εκτός bundle. Τίποτα από αυτά δεν ενσωματώνεται στο signed application directory.
4. **Native tray χωρίς νέο runtime framework.** Χρησιμοποίησε `Shell_NotifyIconW` σε dedicated message loop, `NIM_SETVERSION`, επαναφορά μετά από `TaskbarCreated`, status tooltip και μενού Start Voice, Stop Voice, Open Config Folder, Open Log Folder, Exit. Το tray δεν εισάγει semantic logic.
5. **Single instance.** Χρησιμοποίησε non-inheritable `Local\...` named mutex και αναγνώρισε `ERROR_ALREADY_EXISTS`. Κλείσε πάντα το handle. Μην βάζεις username, path ή άλλο προσωπικό δεδομένο στο όνομα.
6. **Lifecycle.** Η εκκίνηση/παύση του voice runtime γίνεται εκτός του UI thread. Το shutdown είναι idempotent, κάνει άμεσο cancellation/purge και έχει bounded join. Αποθηκεύεται μόνο typed error code, ποτέ exception text, transcript ή path.
7. **DLL policy.** Σε frozen Windows process, runtime hook καλεί `SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)` και προσθέτει μόνο το ελεγχόμενο PyInstaller directory με `AddDllDirectory`/`os.add_dll_directory`. Δεν μεταβάλλει `PATH`. CUDA packaging μένει deferred μέχρι native compatibility/licensing verification.
8. **Distribution manifest.** Παρήγαγε sorted JSON manifest με relative paths, sizes και SHA-256. Απαγόρευσε symlinks, absolute/traversal paths, duplicates, missing/unexpected files και hash/size mismatch. Το manifest είναι integrity inventory, όχι υποκατάστατο Authenticode.
9. **Signing gate.** Τεκμηρίωσε SHA-256 Authenticode + RFC 3161 timestamp και `signtool verify /pa`; μην εκτελέσεις signing χωρίς certificate/explicit approval.
10. **Scope exclusions.** Installer/MSIX, autostart, service, updater, telemetry, wake word, AEC, ambient interruption, bundled CUDA DLLs, bundled models και production orchestrator integration μένουν εκτός M3.

## Παραδοτέα

- application paths, safe local event logging και external-mutable-path guard,
- runtime lifecycle controller με fake-testable interfaces,
- guarded Win32 single-instance, DLL hardening και tray adapters,
- `tray`, `dist-manifest` και `verify-dist` CLI commands,
- PyInstaller spec, runtime hook, Windows application manifest, version resource και offline PowerShell build script,
- ανανεωμένο example config, README και native Windows packaging/smoke guide,
- unit/integration tests που τρέχουν χωρίς τρίτα dependencies και native smoke tests με platform skips.

## Acceptance

- Όλα τα παλιά tests παραμένουν πράσινα και το wire protocol δεν αλλάζει.
- Fake tests αποδεικνύουν lifecycle ordering/idempotence, tray command routing/recovery, mutex duplicate handling, DLL-policy calls, redacted logging και strict distribution verification.
- Το repository κάνει compile χωρίς packaging/runtime dependencies.
- Το macOS run δεν ισχυρίζεται Windows, Shell, SAPI, WASAPI, Authenticode ή CUDA verification.
- Η τελική αναφορά διαχωρίζει καθαρά source-complete, fake-verified, native-deferred και signing-deferred αποτελέσματα.
