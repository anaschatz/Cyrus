# Offline Greek input translation v1

The English-response profile keeps the original Greek ASR transcript and its
language metadata unchanged. Immediately before final local-model context
assembly, Greek-script components are projected to English by a dedicated,
local CPU worker. Current input, the retained history pair, and memory recalls
are translated together; canonical storage, memory search, deterministic
commands, and the Voice protocol continue to use the original text.

## Frozen runtime

- Model: `Helsinki-NLP/opus-mt-tc-big-el-en`
- Revision: `a69108562775c6727edf3540346f9d8900b73634`
- Runtime: CTranslate2 `4.8.1`, SentencePiece `0.2.2`, CPU INT8
- Runtime pack: five closed, hashed artifacts totaling `243579143` bytes
- License: CC-BY-4.0; ship `orchestrator/translation/NOTICE.txt`
- Build provenance: `orchestrator/translation/translation-build-lock.json`

The worker is persistent and has one inference queue. It translates components
as individual examples (`max_batch_size=1`) so adding history or recalls cannot
change the meaning of the current turn through batch-shape numerical drift.
Cancellation owns and terminates the worker process; the replacement turn gets
a new process and stale output is rejected.

Machine values are masked before NMT and restored exactly afterward. The v1
contract covers ASCII URLs, macOS/POSIX file paths, Windows drive and UNC paths,
filenames, times, dates, numbers and units, assignments, hotkeys, acronyms, and
model/product names. Paths containing Greek characters fail closed because the
English-only context gate intentionally rejects Greek script outside the source
transcript.

## Configuration

Use absolute private paths and the exact runtime-manifest SHA-256:

```json
{
  "response_language": "en",
  "input_translation_mode": "opus_ct2",
  "input_translation_model_root": "/absolute/private/path/model",
  "input_translation_manifest_file": "/absolute/private/path/manifest.json",
  "input_translation_manifest_sha256": "c9f7fec27376fea4be404e674d18ef98e3222147e40c27373ceac6dda336e7cc",
  "input_translation_timeout_seconds": 5.0
}
```

Translation is fail-closed. If the model, manifest, worker, timeout, output
schema, English-script check, or protected-span round trip fails, no model
request, `response.started`, or TTS output is emitted. Latin-only English input
bypasses the Greek translator even when the session language hint is `el`.
Tools, cloud routing, and research remain disabled for this English-only v1
projection until every earlier semantic model request consumes the same safe
projection.

## Automated qualification

`orchestrator/scripts/run_translation_qualification.py` requires the model
root, runtime manifest and hash, build lock, development corpus, frozen holdout,
an installed worker interpreter, and a new private output path. The release run
must be executed with network access denied externally.

The current M3 gate contains:

- 82 development/golden translations, including strict safety-critical output;
- 25 holdout paraphrases evaluated after rewrite rules were frozen;
- 250 single turns and three deterministic passes;
- an eight-component current/history/recall context;
- exact protected spans and worker/package identity;
- hard cancellation, PID replacement, fresh-worker determinism, and no orphan;
- cold, warm p95/max, production-batch, replacement, RSS, and RSS-growth limits.

This automated gate deliberately stops before microphone or user testing. A
human voice turn remains a separate physical qualification step.
