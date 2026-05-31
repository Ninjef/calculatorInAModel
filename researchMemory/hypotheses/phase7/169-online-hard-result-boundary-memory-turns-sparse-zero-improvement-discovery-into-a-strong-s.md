# Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source but misses handoff.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-result-boundary.md

Summary:

- Added `--result-boundary-target-online-hard-memory`, which scores sparse result-boundary candidates online, keeps each prompt's best discovered answer-improving result as a hard target, and can freeze rescoring when every prompt has a target. On the op19 full-grid zero-improvement source gate with topk8+unique24 candidates, the 200-step branch only matched the old soft sparse target (`0.455` calc / `0.435` final versus old `0.4275` / `0.4300`), but the 800-step branch matured to `0.9675` learned calc and `0.9725` final. The freeze-when-full variant reached the same source result while stopping forced-result scoring after `86,400` cumulative forced evals instead of about `7,689,600`; the memory was full and `best_true=1.000` by step 50. However, the trusted frozen additive handoff from that source reached only `0.465` final / `0.485` step-600 normal, with calculator accuracy still `0.9575` and injection-zero `0.0100`. This is a strong sparse fixed-grid source mechanism, but it does not yet solve non-bottleneck transfer.

Questions:

- What did we learn about Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source but misses handoff?
- Has Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source but misses handoff been tested?
- Should we repeat Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source but misses handoff?
- What is the status of Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source but misses handoff?
- What follow-up is allowed for Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source but misses handoff?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-result-boundary.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run same-seed op19 online-hard-memory source length/LR repeats as novelty. The useful result is hard online target discovery plus stop-when-full rescoring; the failure mode is handoff/readout geometry.

Next Allowed:

- Add a handoff-aware geometry mechanism to online hard memory, run a fresh-seed source plus trusted handoff, or test streaming/fresh-prompt memory. A source-only repeat is not enough.

Full Text:

```text
MIXED-POSITIVE: Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source but misses handoff.
Conclusion: Added `--result-boundary-target-online-hard-memory`, which scores sparse result-boundary candidates online, keeps each prompt's best discovered answer-improving result as a hard target, and can freeze rescoring when every prompt has a target. On the op19 full-grid zero-improvement source gate with topk8+unique24 candidates, the 200-step branch only matched the old soft sparse target (`0.455` calc / `0.435` final versus old `0.4275` / `0.4300`), but the 800-step branch matured to `0.9675` learned calc and `0.9725` final. The freeze-when-full variant reached the same source result while stopping forced-result scoring after `86,400` cumulative forced evals instead of about `7,689,600`; the memory was full and `best_true=1.000` by step 50. However, the trusted frozen additive handoff from that source reached only `0.465` final / `0.485` step-600 normal, with calculator accuracy still `0.9575` and injection-zero `0.0100`. This is a strong sparse fixed-grid source mechanism, but it does not yet solve non-bottleneck transfer.
Do not repeat: Do not run same-seed op19 online-hard-memory source length/LR repeats as novelty. The useful result is hard online target discovery plus stop-when-full rescoring; the failure mode is handoff/readout geometry.
Next allowed test: Add a handoff-aware geometry mechanism to online hard memory, run a fresh-seed source plus trusted handoff, or test streaming/fresh-prompt memory. A source-only repeat is not enough.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-result-boundary.md`
```
