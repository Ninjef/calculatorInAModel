# A frozen bottleneck-trained result policy can be handed to an additive non-bottleneck model.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-gate.md

Summary:

- Compatible checkpoint loading preserved a `0.9125` bottleneck result policy; without freezing it collapsed to `0.0300` by step `50`, but freezing embeddings/pre-hook block/result head kept final calculator-result accuracy `0.9200` and produced `0.9475` normal versus `0.0175` injection-zero.

Questions:

- What did we learn about A frozen bottleneck-trained result policy can be handed to an additive non-bottleneck model?
- Has A frozen bottleneck-trained result policy can be handed to an additive non-bottleneck model been tested?
- Should we repeat A frozen bottleneck-trained result policy can be handed to an additive non-bottleneck model?
- What is the status of A frozen bottleneck-trained result policy can be handed to an additive non-bottleneck model?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same seed/checkpoint compatible transfer without freezing, or same frozen-policy 800-step handoff as novelty.

Next Allowed:

- Seed/checkpoint replication, staged unfreezing, or a scalable/non-prescriptive way to acquire and preserve the policy.

Full Text:

```text
PARTIAL: A frozen bottleneck-trained result policy can be handed to an additive non-bottleneck model.
Conclusion: Compatible checkpoint loading preserved a `0.9125` bottleneck result policy; without freezing it collapsed to `0.0300` by step `50`, but freezing embeddings/pre-hook block/result head kept final calculator-result accuracy `0.9200` and produced `0.9475` normal versus `0.0175` injection-zero.
Do not repeat: Same seed/checkpoint compatible transfer without freezing, or same frozen-policy 800-step handoff as novelty.
Next allowed test: Seed/checkpoint replication, staged unfreezing, or a scalable/non-prescriptive way to acquire and preserve the policy.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-gate.md`
```
