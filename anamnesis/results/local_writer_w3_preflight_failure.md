# Local W3 writer preflight failure — no scenario run

The bundled W3 memory-compiler repair was tested exactly once with the pinned
local `ollama/qwen3:4b-instruct` model. The nine-call Inspect job had technical
status `success`, but the frozen semantic result was `passed=false`. The
preregistered stopping rule therefore prevented every v4 scenario call: no
experiment manifest was frozen and no writer result table was produced.

This is a development-only compatibility diagnostic. It is not a benchmark,
a single-factor ablation, or evidence for or against the Anamnesis hypothesis.

## Frozen identity

| Item | Value |
|---|---|
| Source/runtime commit | `831dacb1879a055f80d0b469b3bc18f35f308073` |
| Blind v4 dataset commit | `d8b9972` |
| W3 prompt/protocol commit | `a9fb160` |
| Post-prompt fixture commit | `cfe6322` |
| Model | `ollama/qwen3:4b-instruct` |
| Ollama model digest | `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` |
| Compiler prompt SHA-256 | `412a63d6b42ea6b5e294401cabbcbacf5a6b7facddbd8fe04ca7b91914c141e5` |
| Compiler schema SHA-256 | `8871ff344eb3a2e88a53b964ef2f24f089a72507c69073ec323cf26a428c3030` |
| Compiler transport SHA-256 | `57d4c0a6152c5319fcd1adab4071ad010d107f9e65d987c1740fa47adaca1bcc` |
| Decision prompt/schema SHA-256 | `871fe15e3160e66abe7480cbde15dfb943dec2d0ff89bb01a03849ad35defd8d` / `1b7c38d3f4bf150523ecc1e468ad3fb1f94753611f190d70f93abbf5ec582426` |
| Fixture/protocol SHA-256 | `5628c3c1d7f8e1a5da43d6e567d55ac8e4fbabd8b9c4054325de6f4def1da30c` / `7f63c156a8af74ced2d5e5530b3e8083da95c7f46e14e1afafaaf864b3ce1915` |
| Seed / temperature / retries | `101` / `0` / `0` |
| Provider API cost | `$0.00` |

The exact retained preflight is
`results/runs/local/writer_diagnostic_w3/model_preflight.eval`, SHA-256
`f64210de4e7dacc816ded706bdef96228c9276bd3bedcd92e10e0fa4b33ac97e`.
It records clean revision `831dacb`, Ollama `0.31.1`, context length 4096,
Q4_K_M quantization, Metal on Apple M3, the pinned loopback route, and no
cache, retry, repair, or parallel generation.

## Outcome

| Case | Frozen capability | Parse/domain | Semantics | Input | Output | Latency |
|---|---|:---:|:---:|---:|---:|---:|
| C1 | normalized fact key/value/unit | PASS | FAIL | 1,913 | 72 | 16,514.3 ms |
| C2 | bare-weekday `at` reminder | FAIL | FAIL | 1,922 | 161 | 6,426.3 ms |
| C3 | conjunctive condition transition | PASS | FAIL | 1,979 | 31 | 1,745.4 ms |
| C4 | IANA-zone bounded recurrence | FAIL | FAIL | 1,955 | 168 | 6,777.4 ms |
| C5 | exact active-ID trigger update | PASS | PASS | 2,195 | 94 | 4,745.4 ms |
| C6 | full action-template replacement | PASS | FAIL | 2,251 | 165 | 7,594.5 ms |
| C7 | complete sparse payload, explicit zero | PASS | PASS | 1,956 | 274 | 10,808.3 ms |
| C8 | ambiguous reference → empty delta | PASS | FAIL | 2,473 | 67 | 5,052.7 ms |
| D1 | empty structured view → no action | PASS | PASS | 682 | 12 | 2,980.6 ms |

Aggregate usage was 17,326 input and 1,044 output tokens. All usage/cost fields
were complete and provider API cost was exactly zero. Model-call latency was
62,644.9 ms; the residency probe added 4.5 ms. The strict artifact validator
independently rejected the log with
`local W3 semantic preflight result did not pass`.

## Exact failures

- **C1:** the fact entity and attribute were normalized correctly, but the
  model invented unit `percent` although the source gave only the value 73.
- **C2:** it resolved the requested Monday to the wrong week and collapsed the
  source-cased item into the lowercase-only subject. The capitalized subject
  then failed domain validation.
- **C3:** it emitted an empty delta instead of the explicit two-condition
  transition with its bounded active window.
- **C4:** it replaced the requested bounded Tuesday/Saturday recurrence in
  `America/Toronto` with one immediate `at` trigger. Its capitalized subject
  also failed domain validation.
- **C6:** it preserved and changed the action-template leaves correctly, but
  also emitted unchanged `trigger` and `blockers`; the frozen projection
  required an action-template-only update.
- **C8:** the reference was deliberately ambiguous between two active intents;
  instead of failing closed with an empty delta, the model cancelled both.

C5, C7, and D1 passed exactly. In particular, C7 confirms that the W2 sparse
serialization fix and a source-explicit quantity of zero were handled
correctly. The broader W3 semantic bundle was nevertheless incompatible with
this model under the frozen gate.

## Stopping decision

The W3 protocol permits one standalone preflight and authorizes one measured
cell only after a complete pass. That rule was followed:

- writer scenarios evaluated: **0 / 10**;
- writer compiler calls on scenario data: **0 / 39**;
- measured W3 `.eval` logs: **0**;
- W3 experiment manifest frozen: **no**;
- output repair, retry, or selected rerun: **none**;
- W3 candidate status: **rejected at preflight**.

The v4 dataset and reporter-only reference were not used for model evaluation.
Per the frozen rule, no second W3 attempt—and no W4 prompt—may be tuned or run
on these v4 cases. A future diagnostic requires a separately frozen blind
dataset and a new preregistered protocol, or a separately identified model-only
cell that keeps the compiler contract fixed.
