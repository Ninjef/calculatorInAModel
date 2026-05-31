# Additive semantic distillation repairs online-hard-memory handoff geometry on the op19 gate.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-handoff.md

Summary:

- Combined sparse zero-improvement online hard memory with `--additive-semantic-distill-weight 1 --additive-semantic-distill-sample-count 8` during source training. The source still filled/froze memory after `86,400` forced evals, reached final `400/400 = 1.000`, diagnostic calculator-result accuracy `1.000`, and final additive semantic token agreement `0.7459`. The trusted frozen-policy additive handoff from this source reached final `400/400 = 1.000` and step-600 normal `1.000`, with causal controls low (`0.0525` injection-zero, `0.0050` forced-zero, `0.0175` forced-random) and frozen calculator-result accuracy `1.000`. This directly fixes the previous online-hard-memory handoff miss on the same gate without telling the policy which result to request for each prompt.

Questions:

- What did we learn about Additive semantic distillation repairs online-hard-memory handoff geometry on the op19 gate?
- Has Additive semantic distillation repairs online-hard-memory handoff geometry on the op19 gate been tested?
- Should we repeat Additive semantic distillation repairs online-hard-memory handoff geometry on the op19 gate?
- What is the status of Additive semantic distillation repairs online-hard-memory handoff geometry on the op19 gate?
- What follow-up is allowed for Additive semantic distillation repairs online-hard-memory handoff geometry on the op19 gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-handoff.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not tune semantic-distill weight/sample/length on the same op19 seed as novelty. The durable finding is that a non-prescriptive readout-semantics auxiliary can convert the strong sparse source into a handoff-compatible source.

Next Allowed:

- Fresh-seed replication, streaming/fresh-prompt memory, larger-range stress, or routed/many-calculator validation. Also compare whether the semantic-distill auxiliary remains helpful under less fixed-grid memory.

Full Text:

```text
POSITIVE: Additive semantic distillation repairs online-hard-memory handoff geometry on the op19 gate.
Conclusion: Combined sparse zero-improvement online hard memory with `--additive-semantic-distill-weight 1 --additive-semantic-distill-sample-count 8` during source training. The source still filled/froze memory after `86,400` forced evals, reached final `400/400 = 1.000`, diagnostic calculator-result accuracy `1.000`, and final additive semantic token agreement `0.7459`. The trusted frozen-policy additive handoff from this source reached final `400/400 = 1.000` and step-600 normal `1.000`, with causal controls low (`0.0525` injection-zero, `0.0050` forced-zero, `0.0175` forced-random) and frozen calculator-result accuracy `1.000`. This directly fixes the previous online-hard-memory handoff miss on the same gate without telling the policy which result to request for each prompt.
Do not repeat: Do not tune semantic-distill weight/sample/length on the same op19 seed as novelty. The durable finding is that a non-prescriptive readout-semantics auxiliary can convert the strong sparse source into a handoff-compatible source.
Next allowed test: Fresh-seed replication, streaming/fresh-prompt memory, larger-range stress, or routed/many-calculator validation. Also compare whether the semantic-distill auxiliary remains helpful under less fixed-grid memory.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-handoff.md`
```
