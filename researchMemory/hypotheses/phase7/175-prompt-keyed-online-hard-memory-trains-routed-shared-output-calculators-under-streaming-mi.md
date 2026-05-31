# Prompt-keyed online-hard-memory trains routed shared-output calculators under streaming minibatches when exposure is matched.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-prompt-keyed-online-hard-memory-streaming.md

Summary:

- Added `--streaming-train-batch-size` and `--result-boundary-target-online-memory-key-mode prompt`, so sparse zero-improvement online hard memory can train on fresh minibatches instead of requiring the fixed exhaustive-grid batch. On the four-hook `left_operand_mod` routed shared-output op19 gate, batch64 for 800 steps filled/froze all `400` prompt entries with true targets but undertrained the policy (`0.6325` final, diagnostic calculator-result accuracy `0.5781`). The predeclared exposure-matched batch64 source for 5000 steps reached final/calc `1.0000`, filled/froze memory after `173,568` forced evals, and trained all four hooks to calculator-result accuracy `1.0000`. The trusted 600-step frozen-policy additive handoff from that streaming source reached `400/400 = 1.0000` final / step-600 normal, with low controls (`0.0781` final injection-zero, `0.0078` forced-zero, `0.0156` forced-random) and all hooks at calculator-result accuracy `1.0000`.

Questions:

- What did we learn about Prompt-keyed online-hard-memory trains routed shared-output calculators under streaming minibatches when exposure is matched?
- Has Prompt-keyed online-hard-memory trains routed shared-output calculators under streaming minibatches when exposure is matched been tested?
- Should we repeat Prompt-keyed online-hard-memory trains routed shared-output calculators under streaming minibatches when exposure is matched?
- What is the status of Prompt-keyed online-hard-memory trains routed shared-output calculators under streaming minibatches when exposure is matched?
- What follow-up is allowed for Prompt-keyed online-hard-memory trains routed shared-output calculators under streaming minibatches when exposure is matched?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-prompt-keyed-online-hard-memory-streaming.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat the 800-step batch64 miss as a mechanism failure or rerun same-exposure op19 streaming source/handoff as novelty. Do not return to fixed-grid routed/shared op19/op29 repeats.

Next Allowed:

- Fresh/heldout prompt generalization for prompt-keyed memory, or a cheaper streaming uptake mechanism that preserves the matched-exposure source/handoff result with fewer optimizer updates and forced evaluations.

Full Text:

```text
POSITIVE: Prompt-keyed online-hard-memory trains routed shared-output calculators under streaming minibatches when exposure is matched.
Conclusion: Added `--streaming-train-batch-size` and `--result-boundary-target-online-memory-key-mode prompt`, so sparse zero-improvement online hard memory can train on fresh minibatches instead of requiring the fixed exhaustive-grid batch. On the four-hook `left_operand_mod` routed shared-output op19 gate, batch64 for 800 steps filled/froze all `400` prompt entries with true targets but undertrained the policy (`0.6325` final, diagnostic calculator-result accuracy `0.5781`). The predeclared exposure-matched batch64 source for 5000 steps reached final/calc `1.0000`, filled/froze memory after `173,568` forced evals, and trained all four hooks to calculator-result accuracy `1.0000`. The trusted 600-step frozen-policy additive handoff from that streaming source reached `400/400 = 1.0000` final / step-600 normal, with low controls (`0.0781` final injection-zero, `0.0078` forced-zero, `0.0156` forced-random) and all hooks at calculator-result accuracy `1.0000`.
Do not repeat: Do not treat the 800-step batch64 miss as a mechanism failure or rerun same-exposure op19 streaming source/handoff as novelty. Do not return to fixed-grid routed/shared op19/op29 repeats.
Next allowed test: Fresh/heldout prompt generalization for prompt-keyed memory, or a cheaper streaming uptake mechanism that preserves the matched-exposure source/handoff result with fewer optimizer updates and forced evaluations.
Source: `aiAgentWorkHistory/phase7/2026-05-31-prompt-keyed-online-hard-memory-streaming.md`
```
