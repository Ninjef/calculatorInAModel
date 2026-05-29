# Decoder calibration alone rescues ordinary expected-cost discovery.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-14-gradient-friendly-result-decoder-alignment-gate.md

Summary:

- Contrastive-margin decoder passed local sign alignment, then Stage 1 collapsed to wrong low-entropy results.

Questions:

- What did we learn about Decoder calibration alone rescues ordinary expected-cost discovery?
- Has Decoder calibration alone rescues ordinary expected-cost discovery been tested?
- Should we repeat Decoder calibration alone rescues ordinary expected-cost discovery?
- What is the status of Decoder calibration alone rescues ordinary expected-cost discovery?
- Why did Decoder calibration alone rescues ordinary expected-cost discovery fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-14-gradient-friendly-result-decoder-alignment-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Decoder-only sharpening/calibration without a stronger backward channel.

Next Allowed:

- Synthetic gradients, direct feedback alignment, or learned shadow-gradient modules.

Full Text:

```text
DISPROVEN: Decoder calibration alone rescues ordinary expected-cost discovery.
Conclusion: Contrastive-margin decoder passed local sign alignment, then Stage 1 collapsed to wrong low-entropy results.
Do not repeat: Decoder-only sharpening/calibration without a stronger backward channel.
Next allowed test: Synthetic gradients, direct feedback alignment, or learned shadow-gradient modules.
Source: `aiAgentWorkHistory/phase7/2026-05-14-gradient-friendly-result-decoder-alignment-gate.md`
```
