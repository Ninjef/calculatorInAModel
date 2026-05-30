# Shared-output four-hook routed source trains, but trusted handoff misses the cloned-output gate.

Kind: hypothesis_memory
Status: MIXED
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-shared-output-routed-source-handoff.md

Summary:

- Replaced the cloned per-hook output projections in the known four-hook op19 `embd32` topk8+unique24 recipe with `--share-calculator-output-proj`. The source still trained cleanly: final eval `400/400 = 1.0000`, step-630 normal/calc `1.0000`, injection-zero `0.0275`, forced-random `0.0300`, and all four hooks reached calculator-result accuracy `1.0000`. However, the trusted 600-step frozen-policy additive handoff from the tied-output source reached only `0.7625` final eval / `0.7800` step-600 normal, with step-600 calculator-result accuracy `0.9950`, injection-zero `0.0875`, and forced-random `0.0725`. A 600-step continuation from that handoff improved only to `0.7925` final / `0.8050` snapshot normal. A regression/audit verifies that a tied-output checkpoint loaded into tied and independent-hook models gives identical logits, injections, routes, and hook result predictions, so this is not explained by state-dict tying/loading behavior. Shared output projection preserves routed source training and removes parameter growth, but it is not a drop-in replacement for cloned output projections in the trusted non-bottleneck handoff geometry.

Questions:

- What did we learn about Shared-output four-hook routed source trains, but trusted handoff misses the cloned-output gate?
- Has Shared-output four-hook routed source trains, but trusted handoff misses the cloned-output gate been tested?
- Should we repeat Shared-output four-hook routed source trains, but trusted handoff misses the cloned-output gate?
- What is the status of Shared-output four-hook routed source trains, but trusted handoff misses the cloned-output gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-shared-output-routed-source-handoff.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not claim tied output projections have preserved the four-hook non-bottleneck result until a new source/handoff geometry mechanism clears the trusted handoff gate. Do not rerun the same tied-output source630 plus handoff600/continuation600 as novelty.

Next Allowed:

- Diagnose or redesign the transfer geometry for shared-output sources, or move back to less-prescriptive credit assignment. If continuing this branch, require a new mechanism such as handoff-aware source shaping, route-aware downstream readout, or a predeclared tied-output handoff geometry objective.

Full Text:

```text
MIXED: Shared-output four-hook routed source trains, but trusted handoff misses the cloned-output gate.
Conclusion: Replaced the cloned per-hook output projections in the known four-hook op19 `embd32` topk8+unique24 recipe with `--share-calculator-output-proj`. The source still trained cleanly: final eval `400/400 = 1.0000`, step-630 normal/calc `1.0000`, injection-zero `0.0275`, forced-random `0.0300`, and all four hooks reached calculator-result accuracy `1.0000`. However, the trusted 600-step frozen-policy additive handoff from the tied-output source reached only `0.7625` final eval / `0.7800` step-600 normal, with step-600 calculator-result accuracy `0.9950`, injection-zero `0.0875`, and forced-random `0.0725`. A 600-step continuation from that handoff improved only to `0.7925` final / `0.8050` snapshot normal. A regression/audit verifies that a tied-output checkpoint loaded into tied and independent-hook models gives identical logits, injections, routes, and hook result predictions, so this is not explained by state-dict tying/loading behavior. Shared output projection preserves routed source training and removes parameter growth, but it is not a drop-in replacement for cloned output projections in the trusted non-bottleneck handoff geometry.
Do not repeat: Do not claim tied output projections have preserved the four-hook non-bottleneck result until a new source/handoff geometry mechanism clears the trusted handoff gate. Do not rerun the same tied-output source630 plus handoff600/continuation600 as novelty.
Next allowed test: Diagnose or redesign the transfer geometry for shared-output sources, or move back to less-prescriptive credit assignment. If continuing this branch, require a new mechanism such as handoff-aware source shaping, route-aware downstream readout, or a predeclared tied-output handoff geometry objective.
Source: `aiAgentWorkHistory/phase7/2026-05-30-shared-output-routed-source-handoff.md`
```
