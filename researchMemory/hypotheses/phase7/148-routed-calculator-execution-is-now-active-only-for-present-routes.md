# Routed calculator execution is now active-only for present routes.

Kind: hypothesis_memory
Status: POSITIVE-IMPLEMENTATION
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-active-only-routed-hook-execution.md

Summary:

- Updated the model forward path so `calculator_hook_routing='left_operand_mod'` invokes only hooks with examples in the current batch, scatters their traces/injections back into full-batch diagnostics, and reports both configured hooks (`calculator_active_hook_count`) and actually invoked hooks (`calculator_invoked_hook_count`). Updated the routed result-logit helper used by source training so it applies each hook's `result_proj` only to routed examples instead of stacking every hook's logits over the full batch. Regression coverage verifies a four-hook batch routed only to hooks `0` and `2` calls only those hooks, leaves non-routed hook injections zero, and reads result logits only from present routes. This removes the known all-hooks-forward waste from routed batches, but it is an implementation/scaling improvement rather than a new credit-assignment method.

Questions:

- What did we learn about Routed calculator execution is now active-only for present routes?
- Has Routed calculator execution is now active-only for present routes been tested?
- Should we repeat Routed calculator execution is now active-only for present routes?
- What is the status of Routed calculator execution is now active-only for present routes?
- What follow-up is allowed for Routed calculator execution is now active-only for present routes?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-active-only-routed-hook-execution.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not describe the four-hook routed result as still requiring all hooks to execute before route masking after this patch. Also do not claim this solves parameter scaling: cloned/independent output projections still grow with hook count.

Next Allowed:

- Add shared/tied output projections or explicit compute accounting in a routed training run; then return to reduced prescriptiveness or non-enumerative credit assignment rather than more same-seed routed smoke tests.

Full Text:

```text
POSITIVE-IMPLEMENTATION: Routed calculator execution is now active-only for present routes.
Conclusion: Updated the model forward path so `calculator_hook_routing='left_operand_mod'` invokes only hooks with examples in the current batch, scatters their traces/injections back into full-batch diagnostics, and reports both configured hooks (`calculator_active_hook_count`) and actually invoked hooks (`calculator_invoked_hook_count`). Updated the routed result-logit helper used by source training so it applies each hook's `result_proj` only to routed examples instead of stacking every hook's logits over the full batch. Regression coverage verifies a four-hook batch routed only to hooks `0` and `2` calls only those hooks, leaves non-routed hook injections zero, and reads result logits only from present routes. This removes the known all-hooks-forward waste from routed batches, but it is an implementation/scaling improvement rather than a new credit-assignment method.
Do not repeat: Do not describe the four-hook routed result as still requiring all hooks to execute before route masking after this patch. Also do not claim this solves parameter scaling: cloned/independent output projections still grow with hook count.
Next allowed test: Add shared/tied output projections or explicit compute accounting in a routed training run; then return to reduced prescriptiveness or non-enumerative credit assignment rather than more same-seed routed smoke tests.
Source: `aiAgentWorkHistory/phase7/2026-05-30-active-only-routed-hook-execution.md`
```
