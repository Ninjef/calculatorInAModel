# Automated forced-margin recovery survives wider product-decoder parity.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-forced-margin-product-decoder-parity.md

Summary:

- Training a matching `n_embd=32`, `n_head=2`, `answer_decoder_interaction=product` oracle semantic decoder produced a clean scaffold (`1.0000` oracle eval). Using that checkpoint, the automated one-negative forced-margin recovery source improved sharply during the late window (`0.6375 -> 0.9475` source calc from step `600` to `630`) and reached `0.9475` final source eval. The trusted 600-step frozen-policy additive handoff reached `1.0000` final eval / `1.0000` step-600 normal, with injection-zero `0.0000`, forced-random `0.0225` in the step-600 snapshot, and learned calc `0.9700` at step `600` (`0.9297` in the 128-sample summary). This removes the prior wider non-product decoder caveat for the staged benchmark, but remains prescriptive because it still uses hard assignment, true-result forced-margin pressure, a pretrained semantic decoder, and frozen-policy transfer.

Questions:

- What did we learn about Automated forced-margin recovery survives wider product-decoder parity?
- Has Automated forced-margin recovery survives wider product-decoder parity been tested?
- Should we repeat Automated forced-margin recovery survives wider product-decoder parity?
- What is the status of Automated forced-margin recovery survives wider product-decoder parity?
- Why did Automated forced-margin recovery survives wider product-decoder parity fail?
- What follow-up is allowed for Automated forced-margin recovery survives wider product-decoder parity?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-product-decoder-parity.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same wider product-decoder oracle scaffold plus effective-seed-26 automated forced-margin source/handoff as novelty.

Next Allowed:

- Further forced-margin work should stress a genuinely new axis such as larger operand range, larger architecture family, many-calculator cost, or remove hard assignment / true-result forcing with a new target construction or estimator.

Full Text:

```text
POSITIVE: Automated forced-margin recovery survives wider product-decoder parity.
Conclusion: Training a matching `n_embd=32`, `n_head=2`, `answer_decoder_interaction=product` oracle semantic decoder produced a clean scaffold (`1.0000` oracle eval). Using that checkpoint, the automated one-negative forced-margin recovery source improved sharply during the late window (`0.6375 -> 0.9475` source calc from step `600` to `630`) and reached `0.9475` final source eval. The trusted 600-step frozen-policy additive handoff reached `1.0000` final eval / `1.0000` step-600 normal, with injection-zero `0.0000`, forced-random `0.0225` in the step-600 snapshot, and learned calc `0.9700` at step `600` (`0.9297` in the 128-sample summary). This removes the prior wider non-product decoder caveat for the staged benchmark, but remains prescriptive because it still uses hard assignment, true-result forced-margin pressure, a pretrained semantic decoder, and frozen-policy transfer.
Do not repeat: Do not rerun the same wider product-decoder oracle scaffold plus effective-seed-26 automated forced-margin source/handoff as novelty.
Next allowed test: Further forced-margin work should stress a genuinely new axis such as larger operand range, larger architecture family, many-calculator cost, or remove hard assignment / true-result forcing with a new target construction or estimator.
Source: `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-product-decoder-parity.md`
```
