# A hidden result head rescues op29 forced-margin range transfer.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-op29-hidden-result-head-capacity-diagnostic.md

Summary:

- Adding `--calculator-result-head-hidden-size 64` to the op29 product forced-margin source changed the source-capacity picture. With the same op29 oracle decoder and automated one-negative forced-margin schedule, source calc reached `0.9978` at step `630` and final source eval was `0.9978`, versus `0.7133` for the shallow op29 source and `0.8233` after shallow low-LR recovery. The trusted 600-step frozen-policy additive handoff from the `rhead64` step-630 checkpoint reached `1.0000` final eval / `1.0000` step-600 normal, with low controls (`0.0244` injection-zero, `0.0156` forced-random at step `600`) and learned calc `0.9967`. This shows the op29 range failure was strongly source-capacity sensitive, but the method remains prescriptive and full-grid.

Questions:

- What did we learn about A hidden result head rescues op29 forced-margin range transfer?
- Has A hidden result head rescues op29 forced-margin range transfer been tested?
- Should we repeat A hidden result head rescues op29 forced-margin range transfer?
- What is the status of A hidden result head rescues op29 forced-margin range transfer?
- Why did A hidden result head rescues op29 forced-margin range transfer fail?
- What follow-up is allowed for A hidden result head rescues op29 forced-margin range transfer?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-op29-hidden-result-head-capacity-diagnostic.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same op29 `rhead64`, effective-seed-29 source-plus-handoff as novelty.

Next Allowed:

- Test whether this capacity fix survives fresh seeds/larger ranges or can be paired with cheaper assignment; otherwise prioritize removing hard assignment / true-result forcing.

Full Text:

```text
POSITIVE: A hidden result head rescues op29 forced-margin range transfer.
Conclusion: Adding `--calculator-result-head-hidden-size 64` to the op29 product forced-margin source changed the source-capacity picture. With the same op29 oracle decoder and automated one-negative forced-margin schedule, source calc reached `0.9978` at step `630` and final source eval was `0.9978`, versus `0.7133` for the shallow op29 source and `0.8233` after shallow low-LR recovery. The trusted 600-step frozen-policy additive handoff from the `rhead64` step-630 checkpoint reached `1.0000` final eval / `1.0000` step-600 normal, with low controls (`0.0244` injection-zero, `0.0156` forced-random at step `600`) and learned calc `0.9967`. This shows the op29 range failure was strongly source-capacity sensitive, but the method remains prescriptive and full-grid.
Do not repeat: Do not rerun the same op29 `rhead64`, effective-seed-29 source-plus-handoff as novelty.
Next allowed test: Test whether this capacity fix survives fresh seeds/larger ranges or can be paired with cheaper assignment; otherwise prioritize removing hard assignment / true-result forcing.
Source: `aiAgentWorkHistory/phase7/2026-05-30-op29-hidden-result-head-capacity-diagnostic.md`
```
