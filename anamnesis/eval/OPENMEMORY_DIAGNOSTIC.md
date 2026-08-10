# OpenMemory recall diagnostic v1

This is a frozen, development-only decision-layer diagnostic. The OpenMemory
strategy and its prompt boundary were committed at source commit
`96a546d4df2ca65544ec256810c32d43b91b970f` before these cases were authored.
No model call or live OpenMemory call occurred before the dataset freeze.

The artifact contains eight isolated cases: one useful reference-resolution
opportunity, six cases where stale, irrelevant or adversarial recall must not
influence the decision, and one no-hit control. It evaluates only the additive
decision-prompt recall surface. It does not evaluate OpenMemory as an
authoritative temporal store, and it cannot support a final or hypothesis
claim.

## Frozen policy

- Current observable context always overrides recalled text.
- Recall can help resolve a reference in an explicit current action request.
- Recall alone cannot create, cancel, complete or execute an action.
- Fixture/provider IDs can never become action keys or evidence IDs.
- Prompt-injection text inside recall must be treated as data.
- Action summary wording is noncanonical; mode, action key, payload and ordered
  evidence are canonical.
- The evaluated OpenMemory arm is search-only and starts from a fresh,
  independently pinned snapshot.
- Provider-neutral OpenMemory token and cost accounting is unavailable, so the
  arm remains diagnostic and incomplete for usage/cost claims.

## Frozen gate

Run exactly one paired matrix with the same decision model and configuration:

1. decision prompt without retrospective recall;
2. the same prompt with the frozen recall hits.

Promotion requires a gain on the single helpful opportunity, no loss on the
no-hit control, zero additional false actions across all six forbidden-
influence cases, and zero recalled/provider identifiers in action evidence.
All raw outputs and incomplete OpenMemory accounting must be reported. A failed
gate is preserved as a result; the prompt must not be tuned again on these v1
cases.

## Frozen hashes

| Artifact | SHA-256 |
| --- | --- |
| `eval/openmemory/decision_diagnostic.v1.json` bytes | `a1541939dc977ddf233395318ac8470ca17d0bb39ef3284fbd65411edf89e36a` |
| Canonical artifact | `b8da030f0e632c5e85523e75ba9ff948c85950435f8d55ca1b0aa3381e830126` |

The adjacent manifest freezes every record hash, family count, boundary and
stopping rule. Human review remains pending.
