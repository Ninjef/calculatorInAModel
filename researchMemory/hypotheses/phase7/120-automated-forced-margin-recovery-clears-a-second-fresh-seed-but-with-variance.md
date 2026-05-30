# Automated forced-margin recovery clears a second fresh seed but with variance.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-forced-margin-second-fresh-seed-stability.md

Summary:

- Repeating the automated one-negative forced-margin recovery recipe on CLI seed `19` / effective model seed `21` replicated the late recovery mechanism and cleared the trusted handoff gate, but below the prior very strong seed. Source calc rose from `0.5625` at step `600` to `0.8325` at step `630`, and final source eval was `0.8600`. The trusted 600-step frozen-policy additive handoff reached `0.8975` final eval / `0.9050` step-600 normal with injection-zero `0.0000`, forced-random `0.0350`, and learned calc `0.8425` at step `600`. This confirms the recipe is a useful staged-transfer benchmark with real seed variance, not a solved final method.

Questions:

- What did we learn about Automated forced-margin recovery clears a second fresh seed but with variance?
- Has Automated forced-margin recovery clears a second fresh seed but with variance been tested?
- Should we repeat Automated forced-margin recovery clears a second fresh seed but with variance?
- What is the status of Automated forced-margin recovery clears a second fresh seed but with variance?
- Why did Automated forced-margin recovery clears a second fresh seed but with variance fail?
- What follow-up is allowed for Automated forced-margin recovery clears a second fresh seed but with variance?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-second-fresh-seed-stability.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same CLI seed-19/effective-seed-21 automated forced-margin recovery source plus 600-step handoff as novelty.

Next Allowed:

- Forced-margin work should either test broader stability/scale or remove prescriptiveness by replacing hard assignment or true-result forcing; do not tune start step, margin, negative count, or recovery length on this setup as novelty.

Full Text:

```text
MIXED-POSITIVE: Automated forced-margin recovery clears a second fresh seed but with variance.
Conclusion: Repeating the automated one-negative forced-margin recovery recipe on CLI seed `19` / effective model seed `21` replicated the late recovery mechanism and cleared the trusted handoff gate, but below the prior very strong seed. Source calc rose from `0.5625` at step `600` to `0.8325` at step `630`, and final source eval was `0.8600`. The trusted 600-step frozen-policy additive handoff reached `0.8975` final eval / `0.9050` step-600 normal with injection-zero `0.0000`, forced-random `0.0350`, and learned calc `0.8425` at step `600`. This confirms the recipe is a useful staged-transfer benchmark with real seed variance, not a solved final method.
Do not repeat: Do not rerun the same CLI seed-19/effective-seed-21 automated forced-margin recovery source plus 600-step handoff as novelty.
Next allowed test: Forced-margin work should either test broader stability/scale or remove prescriptiveness by replacing hard assignment or true-result forcing; do not tune start step, margin, negative count, or recovery length on this setup as novelty.
Source: `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-second-fresh-seed-stability.md`
```
