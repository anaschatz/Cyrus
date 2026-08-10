# OpenMemory recall diagnostic v2 freeze

V2 is a fresh decision-layer diagnostic created after v1 ended as a transport
failure with no completed pair and no metric. It contains no v1 scenario,
event, recall-hit ID, exact authored surface or entity. The artifact was frozen
before the v2 no-thinking transport was written and before any v2 model call.

The eight families and paired gate remain conceptually unchanged: one reference
resolution opportunity, six stale/irrelevant/adversarial safety cases and one
no-hit control. Mode, action key, payload and ordered observable evidence are
canonical; summary is not. Recall cannot create an action by itself or provide
an evidence identifier.

| Artifact | SHA-256 |
| --- | --- |
| `eval/openmemory/decision_diagnostic.v2.json` bytes | `18d69eec94c35c2b750d2ad75f03db8056881405aaeb7a2838fb36d26593de20` |
| Canonical artifact | `7ce91e19d9ca13e6244ea5917c7a3a4a8e499af458b534f90127abedd2bcea61` |

V2 may be run once under a separately committed no-thinking transport. A valid
failure is terminal. V3 would require another fresh dataset. Human review is
pending, so this remains development-only diagnostic evidence.
