# Frozen handoff replicates for a strong source checkpoint but is sensitive to source checkpoint quality.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-replication.md

Summary:

- Strong source `src2` transferred to additive seeds `2/4` with final eval `0.9400/0.9525` and learned calc `0.9200/0.9150`; weaker sources `src4/src5` preserved learned calc around `0.80-0.87` but reached only `0.3025-0.5550` final eval by 800 steps.

Questions:

- What did we learn about Frozen handoff replicates for a strong source checkpoint but is sensitive to source checkpoint quality?
- Has Frozen handoff replicates for a strong source checkpoint but is sensitive to source checkpoint quality been tested?
- Should we repeat Frozen handoff replicates for a strong source checkpoint but is sensitive to source checkpoint quality?
- What is the status of Frozen handoff replicates for a strong source checkpoint but is sensitive to source checkpoint quality?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-replication.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same frozen 800-step matrix cells `src2_add2`, `src2_add4`, `src4_add2`, `src4_add4`, or `src5_add5` as novelty.

Next Allowed:

- Source checkpoint selection/quality metrics, longer or stronger downstream readout adaptation, staged unfreezing, or a less prescriptive source-policy training method.

Full Text:

```text
PARTIAL: Frozen handoff replicates for a strong source checkpoint but is sensitive to source checkpoint quality.
Conclusion: Strong source `src2` transferred to additive seeds `2/4` with final eval `0.9400/0.9525` and learned calc `0.9200/0.9150`; weaker sources `src4/src5` preserved learned calc around `0.80-0.87` but reached only `0.3025-0.5550` final eval by 800 steps.
Do not repeat: The same frozen 800-step matrix cells `src2_add2`, `src2_add4`, `src4_add2`, `src4_add4`, or `src5_add5` as novelty.
Next allowed test: Source checkpoint selection/quality metrics, longer or stronger downstream readout adaptation, staged unfreezing, or a less prescriptive source-policy training method.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-replication.md`
```
