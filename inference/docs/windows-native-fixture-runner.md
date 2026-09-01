# Windows native Inference fixture runner

This is the source-controlled implementation boundary for
`native-fixture-runner.exe`. It is a separate console executable built from
`inference/scripts/windows_native_fixture_runner.py`; it is not the Voice
package, not an Inference service, and not qualification evidence by itself.

## Build and seal

On the provisioned native Windows source tree, run:

```powershell
.\.venv\Scripts\python.exe -I .\scripts\windows_build_guard.py create-cuda-manifest `
  --root C:\CYRUS\private\cuda-runtime `
  --output C:\CYRUS\private\cuda-runtime-manifest.json

powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 `
  -Stage QualificationRunners `
  -CudaRuntimeManifest C:\CYRUS\private\cuda-runtime-manifest.json `
  -RunnerOutputDirectory C:\CYRUS\staging\qualification-runners
```

The build emits:

```text
C:\CYRUS\staging\qualification-runners\native-fixture-runner.exe
C:\CYRUS\staging\qualification-runners\cyrus-windows-asr-collector.exe
C:\CYRUS\staging\qualification-runners\cyrus-windows-primary-profile-runner.exe
C:\CYRUS\staging\qualification-runners\qualification-runners-manifest.json
```

All three outputs are single-file executables; there is no unbound `_internal`
runtime tree. Preserve the four-file build set immutably outside the Inference
runtime. Copy only the generic runner, ASR collector and primary-profile runner
to their fixed `qualification/` paths in the sealed Inference runtime, then
regenerate its strict manifest; an unrecognized runner-build manifest inside
that exact runtime tree is not permitted. The private handoff record separately
binds the runner-build manifest and its CUDA-manifest digest. Do not substitute
a Python script, a macOS runner, an unsealed EXE, or an artifact from a
different build. Each qualifier hashes the runner it invokes before and after
execution and binds the exact size/digest into the authoritative report.

`cyrus.windows-cuda-runtime-bundle/1` is a canonical, owner-private manifest
outside the DLL directory. It records the exact local fixed-drive root and a
case-insensitively sorted list of `{name,size_bytes,sha256}` entries. The
directory may contain only the declared regular non-reparse DLLs. At minimum it
must contain `cublas64_12.dll`, `cublasLt64_12.dll`, `cudnn64_9.dll` and every
cuDNN 9 split library (`adv`, `cnn`, both engine libraries, `graph`,
`heuristic`, and `ops`). `nvcuda.dll` and `nvml.dll` are forbidden: the NVIDIA
driver supplies them through the safe Windows system search. The builder never
uses `PATH`, `CUDA_PATH`, a toolkit directory, an undeclared wheel DLL, or a
download fallback. Additional entries are limited to recognized CUDA-family
DLL names (plus `zlibwapi.dll`); a system-shadowing name such as `version.dll`
is rejected even when its hash is declared.

The ASR spec removes partial CUDA/cuDNN DLLs found in the CTranslate2 wheel and
embeds only this verified bundle at the onefile extraction root. Its entrypoint
loads every required DLL by exact frozen-root path before importing
CTranslate2. The build starts all three onefile runners with an empty inherited
environment except `SystemRoot`, `WINDIR` and a fresh local `TEMP`/`TMP`; the
expected content-free missing-contract exit proves their boot paths, including
the sealed CUDA loader, without creating qualification evidence. The resulting
`cyrus.windows-qualification-runners/1` manifest binds all three EXE hashes and
the CUDA-manifest hash. It is written once and must travel beside the runners.
This invalid-environment boot smoke proves packaging/loadability only. It does
not load the Whisper model, transcribe audio, exercise CUDA kernels, or qualify
the GPU. The later full collector run still executes with `PATH` stripped and
must pass the complete model-load, 200-clip, fallback and coexistence gates.

Both Inference runner bundles execute the same inference-owned, stdlib-only
runtime hook before application imports. On Windows it selects
`LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` and registers only the frozen extraction
directory; it does not consult a user DLL path and does not import the Voice
package.

## Private ASR corpus intake

The repository contains only the source template
`inference/fixtures/windows-asr-audio-corpus-v1.template.json`. It has no
audio, transcripts, output text, host paths, or claimed pass. The private
corpus root must contain exactly these files:

```text
audio\el\asr-el-001.wav … audio\el\asr-el-100.wav
audio\en\asr-en-001.wav … audio\en\asr-en-100.wav
references\el\asr-el-001.txt … references\el\asr-el-100.txt
references\en\asr-en-001.txt … references\en\asr-en-100.txt
```

Each audio file is PCM WAV, 16 kHz, mono, signed 16-bit, and 1.5–30 seconds.
Each matching reference is strict UTF-8 without BOM, already NFC, 1–4096
bytes, at most 128 normalized words and contains no Unicode control code point.
Build a private manifest beside those files, once, offline:

```powershell
.\inference\.venv\Scripts\python.exe -I .\inference\scripts\build_windows_asr_audio_corpus.py `
  --corpus-root C:\CYRUS\inference\qualification `
  --template .\inference\fixtures\windows-asr-audio-corpus-v1.template.json `
  --output C:\CYRUS\inference\qualification\audio-corpus-v1.json
```

The resulting manifest contains only case ID, locale, safe relative
WAV/reference paths, byte sizes, SHA-256 values, duration and reference word
count. It fixes 100 Greek and 100 English clean cases in case-ID order. The
exact `001..050` cases per locale form the 100-case coexistence subset with a
two-second inference lead-in from LLM start to audio start. Store the manifest,
WAVs and sidecars under owner-only DACLs. Never add reference or hypothesis
text to the manifest, evidence, diagnostics or logs. Short (<1.5 s), ambiguous
and mixed-language material belongs only in the separate non-authoritative
pending report and cannot count toward these clean gates.

Every qualification asset, corpus, model and output must use a local
drive-letter path. UNC shares and Windows device/extended path namespaces are
rejected, and `GetDriveTypeW` rejects mapped network drives, so the offline
qualification cannot read a remote share.

The fixture's `prompt_workload_units` are deterministic repetitions of a fixed
synthetic fragment. They are workload units, not a claim about the exact token
count produced by a particular tokenizer.

## Collector protocol

The fixture runner validates the fixture, runner-adjacent collector, corpus,
power profile and ASR configuration before executing anything. It invokes the
separately sealed `qualification/cyrus-windows-asr-collector.exe` with no
arguments. Its fresh, private child environment contains only the loopback
gateway endpoint/bearer, verified asset paths, and the exact non-reparse
Windows-directory `System32\nvidia-smi.exe` and `powercfg.exe` paths. The
Windows directory comes from `GetWindowsDirectoryW`, not an operator-controlled
environment value. Arbitrary inherited `PATH` is removed; both probes execute
by absolute path. The collector
writes exactly one content-free JSON object to
`CYRUS_ASR_QUALIFICATION_OUTPUT` with these
numeric/identity fields:

```text
asr_runtime_identity_digest, audio_corpus_digest,
greek_utterances, english_utterances,
greek_language_correct, english_language_correct,
greek_reference_words, greek_word_errors,
english_reference_words, english_word_errors,
forced_cpu_fallbacks, overlapping_utterances,
baseline_p95, coexistence_p95, baseline_llm_p95, coexistence_llm_p95,
minimum_free_vram_mib, lost_final_transcripts,
duplicate_final_transcripts, stale_results, oom, crashes, throttle_reasons
```

It does not write audio, transcripts, prompts, model output, raw tool data,
paths, secrets, diagnostics or logs to that output. The runner rejects unknown
or content-shaped fields, mismatched ASR/corpus digests, incorrect 100/100
counts, invalid metrics, a changed asset, non-zero collector exit, timeout, or
any unsafe/reparse path. The parent qualifier applies the final p95,
overlap/headroom, zero-OOM/crash/throttle/lost-transcript and boot-identity
gates. A stale, mismatched, corrupt, or independently generated file with
favorable numbers cannot bypass those checks. The reports explicitly declare
`attestation_level:"identity-bound-local-operator"` and
`operator_trust_required:true`: without hardware-backed signing, a deliberate
same-user actor that coordinates all evidence and matching hashes remains
outside the v1 threat model.

## Generic protocol coverage

Before the collector starts, the runner uses the qualifier's ephemeral
loopback gateway and boot-scoped bearer to run the 20 Greek + 20 English
streaming, capability-dependent named/required tool, strict JSON schema,
cancellation, isolated crash/restart and isolation gates. Prompts and responses
remain process memory only. The final evidence contains only fixed counts,
digests, times and numeric metrics; its failure diagnostic is only a fixed
`stage`/`kind` pair.

## Primary-profile runner boundary

`cyrus-windows-primary-profile-runner.exe` is a second, independently sealed
console executable. Its reviewed implementation is split between
`jarvis_inference.windows_primary_runner` (fixture orchestration) and
`jarvis_inference.windows_primary_metrics` (strict loopback client, Windows
working-set sampling and robust growth evaluation); the script entrypoint and
PyInstaller spec contain no qualification logic. It receives only the
qualifier-provided boot-scoped loopback URL, bearer, challenge and absolute
input/output paths through its private child environment. It accepts no
command-line evidence or operator-provided metric values, and the qualifier
does not forward the user's `PATH` to this absolute sealed executable.

The runner executes the exact 120-case quality corpus and 100-case performance
fixture. For each performance request it first records only the native rendered
token count and SHA-256 of the canonical token-ID array. It then arms one
monotonic generation serial, sends the real public completion request, and
atomically consumes that serial's qualification telemetry. The telemetry is
available only in the ephemeral profile-qualification gateway and contains no
text or token IDs: it binds prompt and accepted-token digests/counts, request
TTFT measured from the gateway request start, backend prompt/decode durations,
accepted-token throughput, and request-to-engine-final elapsed time. Normal
Inference production has no corresponding methods, so those private routes are
404 outside this mode. The normalized 128-token latency is recomputed as
`request TTFT + 127 / accepted decode tokens per second`; short completions are
never mislabeled as observed 128-token responses.

The soak sends the same frozen 100 requests in order for exactly 96 cycles over
at least eight monotonic hours. It performs one deliberate managed-process
restart every 12 cycles, measures each of the eight restart-to-ready durations,
and proves replacement with transient process identities. Evidence retains
only ordinal runner epoch serials, never OS PIDs. Before each restart the runner
has collected 12 combined gateway-plus-engine working-set samples for that
epoch. Only aggregate baseline/final/peak, per-epoch sample counts, robust
global growth and maximum robust epoch growth are persisted; transient peaks
are descriptive and the growth gate uses first/last medians plus a Theil-Sen
slope. The five zero counters in the report are limited to events directly
observed by the runner: request failures, unexpected process termination,
request timeout, cleanup failure and circuit-breaker rejection. It makes no
claim about WDDM eviction, CPU offload or any other unavailable sensor.

The primary evidence also cross-binds the already-qualified generic Windows
ASR/coexistence report, including its fixture, collector, configuration, corpus,
power-profile and ASR-runtime identities. The runner writes one new
challenge-bound content-free evidence file using exclusive atomic creation.
Prompts, responses, transcripts, audio, bearer values, token IDs and host paths
are never written to evidence or diagnostics; a failure diagnostic contains
only a fixed stage and failure kind.

This runner is intentionally still **pending native qualification**. Building
it, generating a corpus manifest, or passing portable fake tests never claims
that the PC, ASR collector, Gemma workload or whole system is ready.
