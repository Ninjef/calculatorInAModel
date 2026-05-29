# The 600-step handoff probe rescues weak `src4` source selection.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-src4.md

Summary:

- Reproduced `src4` with snapshots; 600-step probe selected step `1200` (source normal `0.7550`) over final (`0.8700`), and full frozen handoff improved from old final-source `0.3025` to `0.7800`.

Questions:

- What did we learn about The 600-step handoff probe rescues weak `src4` source selection?
- Has The 600-step handoff probe rescues weak `src4` source selection been tested?
- Should we repeat The 600-step handoff probe rescues weak `src4` source selection?
- What is the status of The 600-step handoff probe rescues weak `src4` source selection?
- What follow-up is allowed for The 600-step handoff probe rescues weak `src4` source selection?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-src4.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same `src4` step `1000/1200/final`, additive seed `2`, frozen-policy handoff-probe comparison as novelty.

Next Allowed:

- Use probe score during source acquisition, reduce probe cost, or test whether probe-selected sources reduce later anchor/long-adaptation needs.

Full Text:

```text
POSITIVE: The 600-step handoff probe rescues weak `src4` source selection.
Conclusion: Reproduced `src4` with snapshots; 600-step probe selected step `1200` (source normal `0.7550`) over final (`0.8700`), and full frozen handoff improved from old final-source `0.3025` to `0.7800`.
Do not repeat: Same `src4` step `1000/1200/final`, additive seed `2`, frozen-policy handoff-probe comparison as novelty.
Next allowed test: Use probe score during source acquisition, reduce probe cost, or test whether probe-selected sources reduce later anchor/long-adaptation needs.
Source: `aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-src4.md`
```
