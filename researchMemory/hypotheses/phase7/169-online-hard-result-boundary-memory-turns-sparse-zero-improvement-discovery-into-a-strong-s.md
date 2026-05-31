# Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source.

Kind: hypothesis_memory
Status: PARTIAL-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-result-boundary.md

Summary:

- Added `--result-boundary-target-online-hard-memory`, which scores sparse result-boundary candidates online, keeps each prompt's best discovered answer-improving result as a hard target, and can freeze rescoring when every prompt has a target. On the op19 full-grid zero-improvement source gate with topk8+unique24 candidates, the 200-step branch only matched the old soft sparse target (`0.455` calc / `0.435` final versus old `0.4275` / `0.4300`), but the 800-step branch matured to `0.9675` learned calc and `0.9725` final. The freeze-when-full variant reached the same result while stopping forced-result scoring after `86,400` cumulative forced evals instead of continuing to about `7,689,600`; the memory was full and `best_true=1.000` by step 50. This is the first sparse answer-derived target construction to match mature source quality without full-enum target refresh, but it is still fixed-grid prompt memory and not yet validated for fresh prompts, additive handoff, or many calculators.

Questions:

- What did we learn about Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source?
- Has Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source been tested?
- Should we repeat Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source?
- What is the status of Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source?
- What follow-up is allowed for Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-result-boundary.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run same-seed op19 online-hard-memory length/LR repeats as novelty. The useful result is hard online target discovery plus stop-when-full rescoring, not another local curve polish.

Next Allowed:

- Validate the mechanism on a fresh seed and a trusted additive handoff, then test streaming/fresh-prompt memory or many-calculator cost. A failure there would mark fixed-grid memory as transductive; a pass would make this the main less-prescriptive scalability branch.

Full Text:

```text
PARTIAL-POSITIVE: Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source.
Conclusion: Added `--result-boundary-target-online-hard-memory`, which scores sparse result-boundary candidates online, keeps each prompt's best discovered answer-improving result as a hard target, and can freeze rescoring when every prompt has a target. On the op19 full-grid zero-improvement source gate with topk8+unique24 candidates, the 200-step branch only matched the old soft sparse target (`0.455` calc / `0.435` final versus old `0.4275` / `0.4300`), but the 800-step branch matured to `0.9675` learned calc and `0.9725` final. The freeze-when-full variant reached the same result while stopping forced-result scoring after `86,400` cumulative forced evals instead of continuing to about `7,689,600`; the memory was full and `best_true=1.000` by step 50. This is the first sparse answer-derived target construction to match mature source quality without full-enum target refresh, but it is still fixed-grid prompt memory and not yet validated for fresh prompts, additive handoff, or many calculators.
Do not repeat: Do not run same-seed op19 online-hard-memory length/LR repeats as novelty. The useful result is hard online target discovery plus stop-when-full rescoring, not another local curve polish.
Next allowed test: Validate the mechanism on a fresh seed and a trusted additive handoff, then test streaming/fresh-prompt memory or many-calculator cost. A failure there would mark fixed-grid memory as transductive; a pass would make this the main less-prescriptive scalability branch.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-result-boundary.md`
```
