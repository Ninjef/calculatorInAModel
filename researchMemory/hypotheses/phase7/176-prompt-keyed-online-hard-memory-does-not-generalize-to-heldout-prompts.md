# Prompt-keyed online-hard-memory does not generalize to heldout prompts.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-prompt-keyed-online-hard-memory-heldout.md

Summary:

- Added a deterministic streaming heldout split and split-specific train/heldout evaluations. On the four-hook shared-output op19 gate, the model trained only on `320` prompts for 5000 batch64 steps, filled/froze exactly those `320` memory entries after `87,552` forced evals, and reached train prompt exact/calc `0.996875` (`319/320`). The `80` heldout prompts, absent from both minibatches and prompt memory, reached only `0.0875` exact/calc (`7/80`), with low controls (`0.0500` injection-zero, `0.0000` forced-zero, `0.0125` forced-random). This is a crisp transductive-memory boundary rather than an optimization failure on seen prompts.

Questions:

- What did we learn about Prompt-keyed online-hard-memory does not generalize to heldout prompts?
- Has Prompt-keyed online-hard-memory does not generalize to heldout prompts been tested?
- Should we repeat Prompt-keyed online-hard-memory does not generalize to heldout prompts?
- What is the status of Prompt-keyed online-hard-memory does not generalize to heldout prompts?
- Why did Prompt-keyed online-hard-memory does not generalize to heldout prompts fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-prompt-keyed-online-hard-memory-heldout.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not launch a trusted handoff or same-exposure repeat from this heldout-failed source as novelty, and do not claim prompt-keyed memory is a fresh-prompt generalization method.

Next Allowed:

- Add a genuinely non-transductive mechanism: amortized target discovery, fresh-prompt candidate scoring/proposal, a learned memory initializer, or another answer-derived credit signal that can produce calculator targets for prompts not already stored.

Full Text:

```text
MIXED-NEGATIVE: Prompt-keyed online-hard-memory does not generalize to heldout prompts.
Conclusion: Added a deterministic streaming heldout split and split-specific train/heldout evaluations. On the four-hook shared-output op19 gate, the model trained only on `320` prompts for 5000 batch64 steps, filled/froze exactly those `320` memory entries after `87,552` forced evals, and reached train prompt exact/calc `0.996875` (`319/320`). The `80` heldout prompts, absent from both minibatches and prompt memory, reached only `0.0875` exact/calc (`7/80`), with low controls (`0.0500` injection-zero, `0.0000` forced-zero, `0.0125` forced-random). This is a crisp transductive-memory boundary rather than an optimization failure on seen prompts.
Do not repeat: Do not launch a trusted handoff or same-exposure repeat from this heldout-failed source as novelty, and do not claim prompt-keyed memory is a fresh-prompt generalization method.
Next allowed test: Add a genuinely non-transductive mechanism: amortized target discovery, fresh-prompt candidate scoring/proposal, a learned memory initializer, or another answer-derived credit signal that can produce calculator targets for prompts not already stored.
Source: `aiAgentWorkHistory/phase7/2026-05-31-prompt-keyed-online-hard-memory-heldout.md`
```
