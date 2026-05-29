# A scheduled additive forced-margin auxiliary can shape source handoff geometry without hurting small-gate source policy acquisition.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-source-aux-gate.md

Summary:

- Adding `--additive-forced-margin-loss-weight` trains the additive path contrastively: the true forced result should have lower answer loss than sampled wrong forced results. On the matched `operand_max=9`, seed-13, 100-step scheduled small gate (`start_step=50`, weight `0.5`, margin `0.05`, 4 negatives), source result-policy accuracy reached `0.4100` and final eval `0.3800`, comparable to the earlier scheduled forced-true small gate (`0.3900`/`0.4000`) and better than always-on forced-true source accuracy (`0.2800`). Geometry improved versus baseline and partly versus scheduled forced-true: `forced_best_true=0.6200`, `top3=0.7500`, and `true-best gap=0.0082`, but 50-step slope final loss `1.0238` was worse than scheduled forced-true (`0.7979`) while still better than baseline (`1.5305`).

Questions:

- What did we learn about A scheduled additive forced-margin auxiliary can shape source handoff geometry without hurting small-gate source policy acquisition?
- Has A scheduled additive forced-margin auxiliary can shape source handoff geometry without hurting small-gate source policy acquisition been tested?
- Should we repeat A scheduled additive forced-margin auxiliary can shape source handoff geometry without hurting small-gate source policy acquisition?
- What is the status of A scheduled additive forced-margin auxiliary can shape source handoff geometry without hurting small-gate source policy acquisition?
- Why did A scheduled additive forced-margin auxiliary can shape source handoff geometry without hurting small-gate source policy acquisition fail?
- What follow-up is allowed for A scheduled additive forced-margin auxiliary can shape source handoff geometry without hurting small-gate source policy acquisition?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-source-aux-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same `operand_max=9`, seed-13, 100-step scheduled forced-margin small gate as novelty.

Next Allowed:

- If pursuing this branch, run a full-grid `operand_max=19` scheduled forced-margin source gate with targeted geometry/handoff validation against the existing scheduled forced-true source objective; otherwise keep source objectives focused on actual 600-step handoff/readout behavior.

Full Text:

```text
MIXED-POSITIVE: A scheduled additive forced-margin auxiliary can shape source handoff geometry without hurting small-gate source policy acquisition.
Conclusion: Adding `--additive-forced-margin-loss-weight` trains the additive path contrastively: the true forced result should have lower answer loss than sampled wrong forced results. On the matched `operand_max=9`, seed-13, 100-step scheduled small gate (`start_step=50`, weight `0.5`, margin `0.05`, 4 negatives), source result-policy accuracy reached `0.4100` and final eval `0.3800`, comparable to the earlier scheduled forced-true small gate (`0.3900`/`0.4000`) and better than always-on forced-true source accuracy (`0.2800`). Geometry improved versus baseline and partly versus scheduled forced-true: `forced_best_true=0.6200`, `top3=0.7500`, and `true-best gap=0.0082`, but 50-step slope final loss `1.0238` was worse than scheduled forced-true (`0.7979`) while still better than baseline (`1.5305`).
Do not repeat: Do not rerun the same `operand_max=9`, seed-13, 100-step scheduled forced-margin small gate as novelty.
Next allowed test: If pursuing this branch, run a full-grid `operand_max=19` scheduled forced-margin source gate with targeted geometry/handoff validation against the existing scheduled forced-true source objective; otherwise keep source objectives focused on actual 600-step handoff/readout behavior.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-source-aux-gate.md`
```
