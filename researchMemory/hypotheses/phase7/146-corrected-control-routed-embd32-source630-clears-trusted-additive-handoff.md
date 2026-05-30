# Corrected-control routed embd32 source630 clears trusted additive handoff.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-routed-embd32-source630-handoff.md

Summary:

- Ran the trusted 600-step frozen-policy additive handoff from the corrected-control fair routed `embd32` source630 checkpoint. The source was the two-hook `left_operand_mod` topk8+unique24 run with cloned output projections and product decoder parity. The handoff reached `400/400 = 1.0000` final eval with final loss effectively zero; the step-600 snapshot had normal `1.0000`, injection-zero `0.0550`, forced-random `0.0300`, oracle `1.0000`, and active-hook calculator-result accuracy `1.0000/0.9955`. Final 128-sample counterfactuals also stayed causal (`0.078125` injection-zero, `0.0234375` forced-random). This is the first corrected-control two-hook routed non-bottleneck staged-transfer positive, so routed sparse assignment is no longer source-only. It remains one seed/op19 and still depends on hard assignment, frozen transfer, cloned per-hook output projections, and a pretrained product decoder.

Questions:

- What did we learn about Corrected-control routed embd32 source630 clears trusted additive handoff?
- Has Corrected-control routed embd32 source630 clears trusted additive handoff been tested?
- Should we repeat Corrected-control routed embd32 source630 clears trusted additive handoff?
- What is the status of Corrected-control routed embd32 source630 clears trusted additive handoff?
- What follow-up is allowed for Corrected-control routed embd32 source630 clears trusted additive handoff?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-routed-embd32-source630-handoff.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun this same effective-seed-43 op19 routed `embd32` source630-to-handoff600 path as novelty. The corrected-control gate is positive.

Next Allowed:

- Move to a thesis-relevant scaling axis: fresh routed seed, more hooks/routes with active-hook cost accounting, or a shared/tied output projection that removes cloned per-hook semantic-output parameter growth.

Full Text:

```text
POSITIVE: Corrected-control routed embd32 source630 clears trusted additive handoff.
Conclusion: Ran the trusted 600-step frozen-policy additive handoff from the corrected-control fair routed `embd32` source630 checkpoint. The source was the two-hook `left_operand_mod` topk8+unique24 run with cloned output projections and product decoder parity. The handoff reached `400/400 = 1.0000` final eval with final loss effectively zero; the step-600 snapshot had normal `1.0000`, injection-zero `0.0550`, forced-random `0.0300`, oracle `1.0000`, and active-hook calculator-result accuracy `1.0000/0.9955`. Final 128-sample counterfactuals also stayed causal (`0.078125` injection-zero, `0.0234375` forced-random). This is the first corrected-control two-hook routed non-bottleneck staged-transfer positive, so routed sparse assignment is no longer source-only. It remains one seed/op19 and still depends on hard assignment, frozen transfer, cloned per-hook output projections, and a pretrained product decoder.
Do not repeat: Do not rerun this same effective-seed-43 op19 routed `embd32` source630-to-handoff600 path as novelty. The corrected-control gate is positive.
Next allowed test: Move to a thesis-relevant scaling axis: fresh routed seed, more hooks/routes with active-hook cost accounting, or a shared/tied output projection that removes cloned per-hook semantic-output parameter growth.
Source: `aiAgentWorkHistory/phase7/2026-05-30-routed-embd32-source630-handoff.md`
```
