# CYRUS Windows primary handoff

Status: **architecture frozen; native Windows qualification pending**.

This is the operator sequence for moving CYRUS from the macOS technical-beta
checkpoint to one native Windows 11 x64 host. It freezes choices and identifies
the evidence boundary; it does not install software, download artifacts, mint a
qualification report, or claim that Windows is ready. If this runbook conflicts
with a linked component contract, the component contract wins.

## Frozen target

1. **One host, one trust boundary.** Voice, Orchestrator, Inference and their
   mutable state run on the same Windows PC under the same non-administrator
   user. Their listeners remain IPv4 loopback-only on `127.0.0.1`; a Mac-to-PC
   split is outside the contract. Mac secrets, databases and readiness evidence
   are not copied to the PC.
2. **Primary model.** `local-primary` is the exact text-only
   `google/gemma-4-12B-it-qat-q4_0-gguf` artifact frozen in
   [model profiles](../inference/docs/model-profiles-v1.md): commit
   `29d097773436b69ff9feafd636ab4cf873786537`, filename
   `gemma-4-12b-it-qat-q4_0.gguf`, `6975879296` bytes, SHA-256
   `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`.
   It uses a pinned Windows `llama-server`, `--no-mmproj`, a 32,768-token
   qualified context cap and the public alias `jarvis-local`.
3. **Controlled challenger, not automatic fallback.** If Gemma fails a frozen
   gate, the primary remains unready. Only then may the operator evaluate
   `Qwen/Qwen3-14B-GGUF`, file `Qwen3-14B-Q4_K_M.gguf`, as a controlled
   challenger. Doing so requires a new explicit profile revision, complete
   artifact lock and full native qualification. Neither the gateway nor the
   Orchestrator may switch to it silently.
4. **ASR order.** Bring-up starts with the existing pinned
   `faster-whisper-base` CPU `int8` profile. The next candidate is the exact
   `large-v3-turbo` pack in
   [the Windows Turbo profile](windows-whisper-large-v3-turbo.md), first loaded
   on CPU `int8`, then qualified on CUDA `int8_float16` with
   `residency: "turn_scoped"`. Warm CUDA residency is a later candidate and is
   forbidden until coexistence passes with the representative Gemma workload.
   The secret-free [RTX 5080 Voice template](../config.windows-rtx5080.template.json)
   records this final candidate shape; its null artifact paths and empty SAPI
   voice descriptions are deliberate and make it non-runnable until native
   materialization.
5. **TTS order.** SAPI5 with explicit installed `el-GR` and `en-US` voice
   descriptions is the baseline. [Pocket TTS](pocket-tts-candidate.md) is an
   optional CPU-only English candidate after the SAPI stack is stable; SAPI5
   remains the Greek and failure fallback. A sentence that has started playing
   is never replayed through another backend.
6. **Bring-up scope.** Memory, Research, Cloud, Tools, Action Host, scheduler
   actions, wake word, autostart and proactive behavior stay disabled until the
   local Voice -> Orchestrator -> qualified Inference path is stable. They are
   not used to make a failed local gate appear successful.

Changing any item above creates a new handoff revision and invalidates evidence
that depends on the changed identity.

## Evidence boundary

| May be completed before moving to Windows | Must be collected on the native Windows host |
|---|---|
| Exact Git commit/tag, clean tracked-tree check and dependency-lock hashes | Windows build/architecture, power profile and real hardware identity |
| Source tests and Windows CI tests that use mocks or ordinary hosted runners | Raw Input, WASAPI, SAPI5, tray, lock/unlock, suspend/resume and device recovery |
| Gemma renderer/parser implementation and deterministic protocol fixtures | Exact rendered-token behavior against the pinned Gemma artifact and runtime |
| Closed Greek/English, structured-output, tool-call and ASR fixture definitions plus their digests | Model quality, performance, VRAM, CUDA/ASR coexistence and eight-hour soak |
| Secret-free Voice, Orchestrator, Inference and private-stack templates | Fresh owner-only/DACL-protected configs, keys, state and evidence |
| Artifact source/revision/license/expected-hash inventory | Byte verification after intake, complete DLL inventory and effective runtime identity |
| Packaging and qualification scripts | Native `onedir` build, Authenticode signing and SignTool verification |
| Mac evidence retained as regression context | Physical PTT, acoustic barge-in, 200-turn Voice evidence and 24-hour system soak |

Source tests, a model card, an estimated VRAM figure, macOS evidence, a copied
report or a successful fake server are never Windows qualification evidence.

## Pre-Windows handoff checklist

The repository is ready to be cloned on the PC only when all of these are true:

- the exact source commit and release/checkpoint tag are recorded, the tracked
  tree is clean, and unrelated local duplicate files are absent from the
  transfer;
- the lock-backed root, Voice, Orchestrator and Inference test commands pass;
- the Gemma renderer, function-call parser and rendered-token/history fixtures
  exist as hashable appliance code; capabilities are not enabled merely by
  editing the manifest; both implementations ship inside the Inference wheel,
  so the handoff does not require separate `runtime/adapters` files;
- the 60 Greek + 60 English primary corpus, 20 + 20 streaming cases, 20 strict
  schemas, named/required tool cases, 100-prompt performance fixture and the
  source-controlled native fixture runner are frozen and digestable; the
  separate ASR corpus has a strict external builder for 200 clean clips (exact
  IDs `001..100` for each of `el-GR` and `en-US`) and matching private UTF-8
  NFC reference `.txt` sidecars. The manifest binds every WAV/reference path,
  size and hash without storing text; the exact first 50 clips per locale form
  the 100-case coexistence subset with a two-second lead-in. Private audio,
  reference text and all native evidence remain deliberately uncommitted;
- secret-free templates exist for `config.example.json`,
  `config.windows-rtx5080.template.json`, `orchestrator/config.example.json`,
  the strict Windows inference manifest and `jarvis-private-stack/2`; all
  optional systems named in Frozen target item 6 default to disabled;
- every required external artifact is either deliberately left for approved
  native intake or has a source/revision/license/size/hash entry in the private
  handoff inventory; this explicitly includes the private canonical CUDA DLL
  bundle manifest and the three-runner build manifest; no cache symlink or
  unpinned alias is accepted;
- the rollback plan names what will be removed, which prior driver/runtime state
  will be restored and how mutable state will be preserved.

This checklist proves that the native work can start. It does not prove that
Gemma, CUDA, Voice or the complete Windows system passes.

### Clean-clone source-run helpers

Three fail-closed PowerShell entry points implement only the preparation and
explicit foreground source-run boundary. Run `provision` first. The two
`run_windows_source_beta` invocations shown next are deliberately deferred
until the generic and profile qualifications in Native execution step 3 pass;
before that, their preflight must fail rather than start an unready gateway:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\provision_windows_source.ps1 `
  -HandoffManifest C:\CYRUS\private\windows-handoff.json

powershell -ExecutionPolicy Bypass -File .\scripts\run_windows_source_beta.ps1 `
  -HandoffManifest C:\CYRUS\private\windows-handoff.json `
  -StackConfig C:\CYRUS\private\cyrus.private.json `
  -Profile CpuBaseline `
  -PreflightOnly

powershell -ExecutionPolicy Bypass -File .\scripts\run_windows_source_beta.ps1 `
  -HandoffManifest C:\CYRUS\private\windows-handoff.json `
  -StackConfig C:\CYRUS\private\cyrus.private.json `
  -Profile CpuBaseline
```

`provision_windows_source.ps1` requires a clean clone, CPython 3.11 x64 and
`uv 0.9.18`. It checks the Voice, Inference and Orchestrator locks and creates
three non-editable environments: root `.venv` with ASR, audio, VAD, transport,
Windows and packaging extras; `inference\.venv` with the exact qualification
extra (`openai==2.46.0`); and `orchestrator\.venv` with its locked test group.
This is the only step that may contact the approved Python package index. Pass
`-Offline` to require a pre-populated uv cache. It sets model-hub offline flags
in both modes and never downloads models, creates keys, materializes configs or
changes machine-wide runtimes. The Inference process is never imported from
`PYTHONPATH`, a tracked egg-info directory or either sibling environment.

`preflight_windows_source.ps1` is runtime-offline and reuses the strict stack
loader, Voice doctor and Inference manifest verifier. It requires configs,
qualification report, Whisper/VAD artifacts and SAPI voice mappings to be
materialized outside the repository; it reads no secret from command-line
arguments and creates or changes none. It also keeps Memory, Research, Cloud,
Tools and scheduler delivery disabled. It validates the private Orchestrator
`inference_bearer_token` but deliberately removes any process-scoped
`JARVIS_INFERENCE_API_KEY` while every preflight subprocess runs. Before the
first tool/doctor/validator process, it snapshots the parent environment and
replaces it with a checked minimal OS environment, the validated tool
directories and only the exact GPU/qualification identity variables. Ambient
proxy, cloud/API and Python injection variables are restored only after
preflight exits.
Because the supervisor probes fixed ports, it also rejects any composition that
is not exactly Voice `127.0.0.1:8765`, Inference `127.0.0.1:8766/v1` and
Orchestrator control `127.0.0.1:8767`; no dynamic-port fallback is attempted.
Operators should normally use `run_windows_source_beta.ps1 -PreflightOnly`, as
shown above: the owned supervisor reads that token from the private config but
does not export it during preflight.
After checking artifacts, the preflight also runs the content-free Inference
`verify-readiness` command. A missing or stale generic/profile report therefore
fails immediately with `windows_source.inference_qualification_not_ready`
instead of waiting for the production supervisor timeout. Qualification itself
continues to use `verify-manifest`, which is intentionally artifact-only.

`run_windows_source_beta.ps1` calls that preflight, starts qualified Inference,
then Orchestrator, then Voice, and waits for authenticated
`jarvis-stack-status/5` readiness. It owns and stops only those three child
processes and their descendants. Every top-level child is created suspended,
assigned to one anonymous Windows Job Object configured with
`KILL_ON_JOB_CLOSE`, and only then resumed; this removes the race where a
parent exits after spawning a helper but before cleanup can discover it. The
Job Object is closed first in `finally`. Each top-level process also retains
the authoritative kernel handle returned by `CreateProcessW`; status, wait and
fallback termination use that handle and never rediscover a process by PID.
Windows CI runs an
orphan smoke in which a parent spawns a sleeper and exits, then requires Job
closure to terminate the sleeper. On Windows PowerShell 5.1 the supervisor
constructs a separate explicit Unicode environment block for each native
`CreateProcessW` call. OS paths come from checked OS APIs, `PATH` is a minimal
checked Windows path, and `TEMP`/`TMP` point at a unique checked runtime child
under the private config directory. That child is removed only after the Job
and retained process handles are drained. Ambient proxy/cloud/Python variables
are not copied. The short status and explicit E2E commands also run under the
sanitized parent environment, not the operator's ambient shell. Only the
Inference block contains the config bearer; neither Orchestrator nor Voice can
inherit it. `-RunE2E`
explicitly requests one content-free real local-model E2E;
without that flag no synthetic request is admitted. `-StopAfterCheck` is for a
bounded connectivity smoke; otherwise Ctrl+C ends the foreground session.
Neither mode installs autostart, a service or a Task Scheduler task, and every
result explicitly says source-run only and not production.

After the CPU baseline passes, repeat preflight/bring-up with
`-Profile CudaTurnScoped` and a privately materialized copy of
`config.windows-rtx5080.template.json`. The helper performs only the passive
doctor; the explicit active CUDA probe and coexistence matrix remain native
qualification steps below.

## Host security prerequisites

Record an explicit operator pass/fail before provisioning:

- supported, fully patched Windows 11 x64; BitLocker enabled for the system and
  CYRUS state volumes; Microsoft Defender and Windows Firewall enabled;
- daily runtime under a dedicated standard user, never Administrator, with no
  elevation request (`asInvoker` package); CPython 3.11 x64 is the only approved
  source/build interpreter;
- all service ports bound only to the frozen numeric loopback endpoints;
  forwarded/origin headers and non-loopback peers remain rejected;
- separate external directories for immutable models/runtime, private config,
  mutable state, logs, backups and qualification evidence;
- owner-only DACLs on configs, authentication material, state, backups and
  evidence; model/runtime artifacts are verified, then made non-writable to the
  runtime user for the qualification session;
- one fresh local `jarvis.voice-auth-key/1`; fresh Orchestrator and Inference
  bearer material; no value in Git, command arguments, logs or evidence. The
  Voice key is file-backed; the Inference key is supplied only to the owned
  service process as required by the appliance contract;
- current-user DPAPI only where a component contract calls for DPAPI; no
  machine-scope fallback and no migration of macOS Keychain material;
- runtime model/tokenizer resolution and downloads disabled. Artifact intake is
  a separate, explicit operator action followed by offline hash verification;
- the native Windows doctor and both CPU/CUDA runtime constructors reject a
  symlink or reparse point at the Whisper pack root or at any pack member; the
  legacy macOS cache-symlink behavior is not accepted for this handoff;
- signing key/PFX and password remain outside the repository, terminal history
  and application directories. Signing occurs only in the controlled release
  step.

Loopback authentication does not encrypt traffic, sandbox a process or defend
against malware already executing as the same Windows user. Host hardening is
therefore a prerequisite, not a replacement for least privilege and bounded
component capabilities.

## Private BOM and handoff record

Keep this inventory outside Git with an owner-only DACL. Sanitized qualification
schemas remain closed and must not be extended with paths, hostnames, device IDs
or notes.

| Record | Required fields |
|---|---|
| Source | repository URL, exact 40-hex commit, checkpoint tag, tracked-tree status, submodule identities if any |
| Lock | relative path, byte size, SHA-256, lock/schema version, generation command/tool version |
| Host | Windows edition/build, x64, power-scheme GUID and AC state, CPU/RAM, RTX 5080 VRAM, GPU driver, GPU power limit/clock policy |
| Compute | `llama-server` commit/build ID, CUDA and cuDNN/runtime versions, CTranslate2 and faster-whisper versions, compute capability |
| Artifact | logical ID, safe relative path, byte size, SHA-256, upstream repository, full revision, license, acquisition approval |
| Python package | canonical name/version, wheel filename, wheel byte size/SHA-256, license, transitive-lock membership |
| Native binary | relative path, byte size/SHA-256, signer/timestamp verification result; complete colocated DLL allowlist |
| CUDA bundle | canonical manifest SHA-256, fixed local non-reparse root, exact DLL names/sizes/SHA-256/licenses; cublas, cublasLt and complete cuDNN 9 closure; no driver DLL |
| Qualification runners | canonical runner-manifest SHA-256, exact three EXE identities and bound CUDA-manifest SHA-256 |
| Model profile | profile schema/ID, artifact digest, renderer/parser/schema-enforcer digests, context/completion caps, sampling/KV/batch/GPU placement |
| Config | Voice/Orchestrator/Inference/stack schema versions and redacted canonical projection digests; never secret values |
| Evidence | gate name/schema, status, artifact/runtime/config identity digests, fixture digest, start/end time, report SHA-256 |
| Rollback | previous approved identity, stop order, state/backup preservation, exact removal/reversion actions, post-rollback checks |

The Windows inference manifest itself must obey
[manifest v1](../inference/docs/manifest-v1.md): every artifact is exactly
`path`, `size_bytes`, and `sha256`; every colocated DLL is declared; Python is
`{}`; unknown fields, symlinks/reparse points and extra model files fail closed.

## Native execution order

Do not skip ahead or reuse evidence across an identity change.

### 1. Inventory, hardening and offline staging

Record the security prerequisites and BOM first. Obtain explicit approval for
the exact driver/CUDA/cuDNN/CTranslate2 matrix before installing it. Keep a
rollback point. Do not modify machine-wide `PATH` and do not download from an
application runtime.

### 2. Source and lock verification

From the exact recorded commit, run `provision_windows_source.ps1` first, then
use only its explicit environments. Run the lock-backed CI commands in
[Backend CI](backend-ci.md), then at minimum:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests inference\src inference\tests orchestrator\src orchestrator\tests
$env:CYRUS_NATIVE_RELEASE_HOST_TEST = "1"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:CYRUS_NATIVE_RELEASE_HOST_TEST
.\inference\.venv\Scripts\python.exe -m unittest discover -s inference\tests -v
.\inference\.venv\Scripts\python.exe .\inference\tests\sdk_conformance.py -v
.\orchestrator\.venv\Scripts\python.exe -m unittest discover -s orchestrator\tests -v
```

A skipped native test is not a pass. The explicit environment flag above is
never set by hosted CI: it makes the Windows 11 release-host gate run only on
the actual target PC, while Server 2022 CI remains a portability/security
smoke and cannot be mistaken for release qualification. Remove the flag again
immediately after the Voice suite. Resolve the exact offline wheels from the
approved locks; do not let these commands select newer dependencies.

### 3. Gemma-only Inference qualification

Before staging the manifest used below, build the three qualification runners
once. First place only the approved CUDA/cuDNN DLL closure in a new owner-only,
fixed-local, non-reparse directory. Generate its canonical private identity and
build into a new output directory:

```powershell
.\.venv\Scripts\python.exe -I .\scripts\windows_build_guard.py create-cuda-manifest `
  --root C:\CYRUS\private\cuda-runtime `
  --output C:\CYRUS\private\cuda-runtime-manifest.json

powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 `
  -Stage QualificationRunners `
  -CudaRuntimeManifest C:\CYRUS\private\cuda-runtime-manifest.json `
  -RunnerOutputDirectory C:\CYRUS\staging\qualification-runners
```

This stage uses isolated PyInstaller invocations, rejects reparse or non-local
inputs, replaces the wheel's partial CUDA payload with only the manifest-bound
DLLs, and minimal-environment boot-smokes all three onefile runners. Keep the
four-file build set together in immutable private staging/evidence, but copy
only the three expected EXEs into the exact Inference `qualification/` paths;
the strict Inference tree does not admit the build manifest as an extra file.
Bind each EXE in the Inference manifest and bind the separate runner/CUDA
manifests in the private handoff record. This boot smoke is packaging evidence,
not model-load/CUDA qualification; the full PATH-stripped collector run remains
mandatory. Never rerun this stage after qualification begins; create a new
runner identity and repeat affected evidence instead.

Materialize the strict Windows manifest and profile lock, including the complete
`llama-server.exe`/DLL inventory and the exact Gemma file. Verify before spawn:

```powershell
.\inference\.venv\Scripts\python.exe -I -m jarvis_inference verify-manifest `
  --manifest C:\CYRUS\inference\manifest.json
```

Set the six `JARVIS_GATEWAY_WHEEL` / `JARVIS_QUAL_GPU_*` identity variables in
the owned PowerShell process as documented in
[the Inference README](../inference/README.md), then run the local, hash-pinned
fixture runner:

```powershell
.\inference\.venv\Scripts\python.exe -I -m jarvis_inference qualify `
  --manifest C:\CYRUS\inference\manifest.json `
  --fixture-runner C:\CYRUS\inference\native-fixture-runner.exe
```

The generic appliance report is necessary but insufficient. The same exact
identity must also pass all `local-primary` quality, structured/tool,
performance, headroom and eight-hour profile gates in
[model profiles](../inference/docs/model-profiles-v1.md). Until then `/readyz`
must remain false and the Orchestrator must not route production work to it.

Stage the two frozen quality corpus files, the frozen performance/soak fixture,
the primary profile runner, and the preserved-evidence destination below the
same sealed manifest root using their exact `qualification/...` contract
paths. Then run the separate challenge-bound profile qualifier:

```powershell
.\inference\.venv\Scripts\python.exe -I -m jarvis_inference qualify-profile `
  --manifest C:\CYRUS\inference\manifest.json `
  --fixture-runner C:\CYRUS\inference\qualification\cyrus-windows-primary-profile-runner.exe

.\inference\.venv\Scripts\python.exe -I -m jarvis_inference verify-readiness `
  --manifest C:\CYRUS\inference\manifest.json
```

The first command uses an ephemeral qualification-only gateway after verifying
the same-identity generic pass. It must run the real 120-case quality suite,
100-case native-tokenized performance fixture, eight-hour/96-cycle soak with
at least eight restarts, and coexistence checks. Only the qualifier writes
`qualification-profile-local-primary.json` in the supported workflow; a
pending fixture is never a pass. The final command emits only readiness,
manifest digest, `attestation_level:"identity-bound-local-operator"`, and
`operator_trust_required:true`. This is not hardware-backed attestation: it
detects stale, mismatched, corrupt and uncoordinated edits, but deliberately
trusts the local service account/operator and cannot resist coordinated
same-user hand-authored evidence.

The source and sealing protocol for `native-fixture-runner.exe`, the private
audio-corpus builder and its collector-only interface are in
[Windows native fixture runner](../inference/docs/windows-native-fixture-runner.md).
The builder fixes metadata only; no PC result is created before this command.

If Gemma fails, preserve the failed report. Do not tune thresholds after seeing
results and do not activate Qwen3-14B. A Qwen evaluation begins only through the
new-profile procedure in Frozen target item 3.

### 4. Voice CPU baseline

Create fresh private config and auth material, keep
`asr.model_profile: "base"`, `asr.execution_profile: "cpu"`, and
`windows_tts_backend: "sapi5"`, then execute
[the native Windows smoke](windows-smoke.md) in order:

```powershell
.\.venv\Scripts\python.exe -I -m jarvis_voice create-auth-key --output C:\CYRUS\private\voice-auth.json
.\.venv\Scripts\python.exe -I -m jarvis_voice doctor --config C:\CYRUS\private\voice.json
.\.venv\Scripts\python.exe -I -m jarvis_voice fake-server --config C:\CYRUS\private\voice.json
.\.venv\Scripts\python.exe -I -m jarvis_voice run --config C:\CYRUS\private\voice.json
```

Run fake server and Voice in separate terminals. Stop the fake server before
starting the real Orchestrator; they never share a listener.

### 5. Real single-host stack

Start qualified Inference, then Orchestrator, then Voice. Materialize fresh
owner-private configs and a `jarvis-private-stack/2` manifest as described in
[Private stack](private-stack-run.md). Require authenticated `status`, one real
local-model `e2e`, the bounded local-model failure gate, restart, another
`status`/`e2e`, and one physical PTT turn. A Windows sealed inference runner and
matching pass report must exist before stack initialization; a macOS runner or
report is ineligible.

### 6. Turbo CUDA `turn_scoped`

Validate the exact five-file Turbo pack, prove CPU `int8` fallback, then run the
passive and active probes:

Materialize a private copy of
[`config.windows-rtx5080.template.json`](../config.windows-rtx5080.template.json),
replacing every required null/empty artifact or voice field with an exact native
value. Never edit the checked-in template with local paths or secrets.

```powershell
.\.venv\Scripts\python.exe -I -m jarvis_voice doctor --config C:\CYRUS\private\voice-turbo.json
.\.venv\Scripts\python.exe -I -m jarvis_voice doctor --config C:\CYRUS\private\voice-turbo.json --active-cuda-probe
```

Run every profile and sample group in
[CUDA coexistence](cuda-coexistence.md) with the exact qualified Gemma workload.
Admission requires 4 GiB observed free at the sample point (2 GiB for Voice and
2 GiB reserved for display/KV/spikes); the profile gate additionally requires
at least 2,048 MiB free throughout overlap. Require zero OOM/crash/lost or
duplicate final/stale result, VRAM return after worker exit and no more than 15%
ASR or LLM p95 degradation under their respective contracts.

Do not test warm ASR residency until `turn_scoped` passes.

### 7. TTS candidate

Complete the SAPI5 physical cases first. Pocket TTS may then be installed only
from its approved lock and exact three artifacts. Promotion requires a separate
RTX 5080-host run: at least 100 warm English cases, p95 first audio no greater
than 600 ms, p5 synthesis at least 2x realtime, physical cancellation below
200 ms, coexistence/restart recovery and an eight-hour zero-growth soak. A
failed Pocket run leaves SAPI5 active and does not block the baseline stack.

### 8. Package, sign, then bind Voice evidence

Build the native windowed `onedir` only after source smoke is stable. This
stage verifies the already-qualified three-runner manifest and cannot rebuild
or mutate those runner bytes:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 `
  -Stage VoicePackage `
  -RunnerBuildManifest C:\CYRUS\staging\qualification-runners\qualification-runners-manifest.json
```

Follow [Windows packaging](windows-packaging.md): inspect licenses/SBOM and the
warning file; sign every PE with the approved SHA-256/RFC 3161 process;
regenerate the distribution manifest after the final binary mutation; require
SignTool exit `0` and `verify-dist` success. Only then create the sample and
native-hardware templates. Any later package change invalidates both.

### 9. Native Voice and release gates

Collect at least 200 physical turns, 20 valid acoustic interruptions, device and
sleep recovery, all Windows platform cases and the eight-hour RSS soak under
[Native Voice qualification](native-voice-qualification.md). Finalize with the
exact commands and distribution binding in
[Native Windows qualification](native-windows-qualification.md):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\qualify_windows.ps1 `
  -Config C:\CYRUS\private\voice.json `
  -Distribution "C:\CYRUS\dist\JARVIS Voice" `
  -Samples C:\CYRUS\evidence\cpu-samples.json `
  -NativeEvidence C:\CYRUS\evidence\native-hardware.json `
  -Output C:\CYRUS\evidence\release-qualification.json
```

The package is eligible only when the generated report says `ready:true` and
the operator approves the recorded latency distributions.

### 10. Whole-system operational gate

With the same qualified identities, collect one closed 72-hour cohort in this
fixed order: 1,000-turn synthetic gate; 100-turn real local-model gate; restart
gate; disposable corruption/backup/restore gate; 24-hour soak; signed native
Voice projection; aggregate evaluation. Use the exact commands and signing-key
custody rules in [Operational Ready Gate v4](../orchestrator/docs/operational-ready-gate-v1.md).

Only the signature-verified aggregate with `ready:true` qualifies the whole
Windows system. Component reports cannot substitute for the aggregate.

## Acceptance summary

| Gate | Minimum decision |
|---|---|
| Gemma quality | 60 Greek + 60 English; at least 90% in each; gap at most 10 points; zero critical failures |
| Structured/tool | 20/20 strict schemas; named/required calls wire-valid and at least 19/20 exact matches |
| Gemma performance | p95 full request TTFT (auth/parse/prepare included) <= 3 s; p95 projected 128-token elapsed (`TTFT + 127/rate`) <= 10 s; p5 >= 25 accepted tokens/s; exact native token bins; 100/100 success |
| Gemma stability | 8 hours, >=96 cycles, >=8 full restarts; zero observed request/process/timeout/cleanup/breaker failures; >=12 combined working-set samples/epoch and <=5% robust within-epoch/global Theil–Sen growth |
| GPU/ASR coexistence | 100 Greek + 100 English clean clips, >=95 language-correct and <=20% WER each; exact first 50/locale matched coexistence subset with >=80 real overlaps; five bilingual injected production CPU-fallback routes; >=2,048 MiB free; no lost/duplicate/stale result; ASR and LLM p95 regression <=15%; zero measured OOM/throttle/crash |
| Voice | >=200 physical turns; zero duplicate/stuck states; >=20 acoustic barge-ins, every sample <200 ms; recovery cases pass; 8-hour RSS gate |
| Package | `onedir`, `asInvoker`, DLL hardening, complete manifest, every PE SignTool exit 0 |
| Whole system | 1,000 synthetic, 100 live, restart and recovery pass, 24-hour soak, matching signed native leaf, signed aggregate `ready:true` |

## Known state at handoff

The repository currently records Gemma as **selected, not qualified** and says
there is no Windows primary bundle or native pass report. The Qwen3-14B file is
only a comparison control. The macOS technical-beta reports cannot qualify the
PC. These are truthful starting conditions for native bring-up, not missing
values to fill with estimates.
