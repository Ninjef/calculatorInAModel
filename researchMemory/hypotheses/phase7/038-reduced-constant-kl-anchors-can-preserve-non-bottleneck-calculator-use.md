# Reduced constant KL anchors can preserve non-bottleneck calculator use.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-reduced-anchor-strength.md

Summary:

- Anchor weights `1.0` and `0.1` at LR `3e-4` kept final calc near `0.77-0.81`, final eval `0.7775/0.9925` for weight `1` and `0.8325/0.9750` for weight `0.1`, with injection-zero near chance.

Questions:

- What did we learn about Reduced constant KL anchors can preserve non-bottleneck calculator use?
- Has Reduced constant KL anchors can preserve non-bottleneck calculator use been tested?
- Should we repeat Reduced constant KL anchors can preserve non-bottleneck calculator use?
- What is the status of Reduced constant KL anchors can preserve non-bottleneck calculator use?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-reduced-anchor-strength.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, anchor weights `1.0` or `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.

Next Allowed:

- Even weaker/floored/gated anchors, selective unfreeze, or source-policy training that reduces the need for an anchor.

Full Text:

```text
PARTIAL: Reduced constant KL anchors can preserve non-bottleneck calculator use.
Conclusion: Anchor weights `1.0` and `0.1` at LR `3e-4` kept final calc near `0.77-0.81`, final eval `0.7775/0.9925` for weight `1` and `0.8325/0.9750` for weight `0.1`, with injection-zero near chance.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weights `1.0` or `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Even weaker/floored/gated anchors, selective unfreeze, or source-policy training that reduces the need for an anchor.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-reduced-anchor-strength.md`
```
