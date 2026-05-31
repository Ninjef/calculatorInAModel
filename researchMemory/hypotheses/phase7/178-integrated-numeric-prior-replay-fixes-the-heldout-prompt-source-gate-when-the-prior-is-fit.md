# Integrated numeric-prior replay fixes the heldout prompt source gate when the prior is fit on full memory.

Kind: hypothesis_memory
Status: POSITIVE-WITH-CAVEAT
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-integrated-amortized-prior-source-gate.md

Summary:

- The first integrated 5000-step source with numeric prior replay but minibatch prior fit improved heldout prompts from `0.0875` to `0.7125`, while train stayed `0.990625`; an offline full-batch prior fit from that final train trace recovered `0.9125` heldout target accuracy, identifying online prior fit quality as the blocker. Adding `--result-boundary-target-amortized-prior-fit-batch-size 0` and rerunning the same op19 four-hook shared-output heldout source reached `398/400 = 0.9950` overall, train `320/320 = 1.0000`, heldout `73/80 = 0.9125`, low heldout controls (`0.0500` injection-zero, `0.0000` forced-zero, `0.0125` forced-random), and online prior heldout accuracy `0.9250` with `86,016` forced evals. The trusted frozen-policy additive handoff from this source reached `400/400 = 1.0000` with low controls (`0.0234` injection-zero, `0.0078` forced-zero, `0.0156` forced-random) and diagnostic calc `0.984375`.

Questions:

- What did we learn about Integrated numeric-prior replay fixes the heldout prompt source gate when the prior is fit on full memory?
- Has Integrated numeric-prior replay fixes the heldout prompt source gate when the prior is fit on full memory been tested?
- Should we repeat Integrated numeric-prior replay fixes the heldout prompt source gate when the prior is fit on full memory?
- What is the status of Integrated numeric-prior replay fixes the heldout prompt source gate when the prior is fit on full memory?
- What follow-up is allowed for Integrated numeric-prior replay fixes the heldout prompt source gate when the prior is fit on full memory?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-integrated-amortized-prior-source-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun same-seed op19 full-memory prior fit as novelty, and do not treat full-memory prior fitting as already scalable.

Next Allowed:

- Reduce prior-fit cost while preserving heldout source accuracy and trusted handoff causality, using cached/periodic full-memory fits, multiple updates only when memory changes, or a coreset/reservoir fit batch before fresh-seed replication.

Full Text:

```text
POSITIVE-WITH-CAVEAT: Integrated numeric-prior replay fixes the heldout prompt source gate when the prior is fit on full memory.
Conclusion: The first integrated 5000-step source with numeric prior replay but minibatch prior fit improved heldout prompts from `0.0875` to `0.7125`, while train stayed `0.990625`; an offline full-batch prior fit from that final train trace recovered `0.9125` heldout target accuracy, identifying online prior fit quality as the blocker. Adding `--result-boundary-target-amortized-prior-fit-batch-size 0` and rerunning the same op19 four-hook shared-output heldout source reached `398/400 = 0.9950` overall, train `320/320 = 1.0000`, heldout `73/80 = 0.9125`, low heldout controls (`0.0500` injection-zero, `0.0000` forced-zero, `0.0125` forced-random), and online prior heldout accuracy `0.9250` with `86,016` forced evals. The trusted frozen-policy additive handoff from this source reached `400/400 = 1.0000` with low controls (`0.0234` injection-zero, `0.0078` forced-zero, `0.0156` forced-random) and diagnostic calc `0.984375`.
Do not repeat: Do not rerun same-seed op19 full-memory prior fit as novelty, and do not treat full-memory prior fitting as already scalable.
Next allowed test: Reduce prior-fit cost while preserving heldout source accuracy and trusted handoff causality, using cached/periodic full-memory fits, multiple updates only when memory changes, or a coreset/reservoir fit batch before fresh-seed replication.
Source: `aiAgentWorkHistory/phase7/2026-05-31-integrated-amortized-prior-source-gate.md`
```
