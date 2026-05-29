# Local-target propagation style result-boundary targets can pass Stage 0, but sharp settings mostly recover the full-enum boundary teacher.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-local-target-propagation-gate.md

Summary:

- A new exact-grid Stage 0 diagnostic found that current-policy-reweighted forced-loss targets align with the hard boundary ceiling when sharp (`t=0.25` result/upstream cosine `~1.0/~1.0`) and remain strongly aligned at `t=1.0` (`0.9355/0.8766`), while ordinary expected answer loss stayed anti-aligned (`-0.1045/-0.0034`). Local logit-descent targets also aligned when weakly proximal (`p=0.01` `~1.0/~1.0`, `p=0.1` `0.9998/0.9997`), but a stronger proximity setting collapsed toward the failed expected-loss direction (`p=1.0` `-0.0895/-0.0028`).

Questions:

- What did we learn about Local-target propagation style result-boundary targets can pass Stage 0, but sharp settings mostly recover the full-enum boundary teacher?
- Has Local-target propagation style result-boundary targets can pass Stage 0, but sharp settings mostly recover the full-enum boundary teacher been tested?
- Should we repeat Local-target propagation style result-boundary targets can pass Stage 0, but sharp settings mostly recover the full-enum boundary teacher?
- What is the status of Local-target propagation style result-boundary targets can pass Stage 0, but sharp settings mostly recover the full-enum boundary teacher?
- Why did Local-target propagation style result-boundary targets can pass Stage 0, but sharp settings mostly recover the full-enum boundary teacher fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-local-target-propagation-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-2 exact-grid Stage 0 sweep over policy-reweighted temperatures `0.25/0.5/1/2` and logit-descent proximity `0.01/0.1/1` as novelty.

Next Allowed:

- If continuing this family, run a short Stage 1 lift gate for the softer aligned settings (`policy_reweighted_t1` or `logit_descent_p0.1`) against the hard-boundary ceiling, then design an approximation that avoids full result-class enumeration before calling it scalable.

Full Text:

```text
PARTIAL: Local-target propagation style result-boundary targets can pass Stage 0, but sharp settings mostly recover the full-enum boundary teacher.
Conclusion: A new exact-grid Stage 0 diagnostic found that current-policy-reweighted forced-loss targets align with the hard boundary ceiling when sharp (`t=0.25` result/upstream cosine `~1.0/~1.0`) and remain strongly aligned at `t=1.0` (`0.9355/0.8766`), while ordinary expected answer loss stayed anti-aligned (`-0.1045/-0.0034`). Local logit-descent targets also aligned when weakly proximal (`p=0.01` `~1.0/~1.0`, `p=0.1` `0.9998/0.9997`), but a stronger proximity setting collapsed toward the failed expected-loss direction (`p=1.0` `-0.0895/-0.0028`).
Do not repeat: The same seed-2 exact-grid Stage 0 sweep over policy-reweighted temperatures `0.25/0.5/1/2` and logit-descent proximity `0.01/0.1/1` as novelty.
Next allowed test: If continuing this family, run a short Stage 1 lift gate for the softer aligned settings (`policy_reweighted_t1` or `logit_descent_p0.1`) against the hard-boundary ceiling, then design an approximation that avoids full result-class enumeration before calling it scalable.
Source: `aiAgentWorkHistory/phase7/2026-05-29-local-target-propagation-gate.md`
```
