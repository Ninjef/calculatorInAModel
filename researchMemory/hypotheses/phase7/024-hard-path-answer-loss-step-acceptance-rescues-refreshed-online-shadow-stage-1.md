# Hard-path answer-loss step acceptance rescues refreshed online-shadow Stage 1.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-answer-loss-step-acceptance-gate.md

Summary:

- Accept/reject gating with tolerances `0.0` and `0.1` accepted only `6/200` proposed steps (`3%`) and ended at `0.050` final exact with best snapshot `0.070`.

Questions:

- What did we learn about Hard-path answer-loss step acceptance rescues refreshed online-shadow Stage 1?
- Has Hard-path answer-loss step acceptance rescues refreshed online-shadow Stage 1 been tested?
- Should we repeat Hard-path answer-loss step acceptance rescues refreshed online-shadow Stage 1?
- What is the status of Hard-path answer-loss step acceptance rescues refreshed online-shadow Stage 1?
- Why did Hard-path answer-loss step acceptance rescues refreshed online-shadow Stage 1 fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-answer-loss-step-acceptance-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same refreshed h32 validation-gradient module with feedback clamp `10`, answer-loss acceptance tolerance `0.0` or `0.1`, and 200-step budget as novelty.

Next Allowed:

- A mechanism that repairs/constructs useful directions rather than simply rejecting most shadow steps, hard assignment-style usage constraints, Jacobian-conditioned state, or richer targets.

Full Text:

```text
DISPROVEN: Hard-path answer-loss step acceptance rescues refreshed online-shadow Stage 1.
Conclusion: Accept/reject gating with tolerances `0.0` and `0.1` accepted only `6/200` proposed steps (`3%`) and ended at `0.050` final exact with best snapshot `0.070`.
Do not repeat: Same refreshed h32 validation-gradient module with feedback clamp `10`, answer-loss acceptance tolerance `0.0` or `0.1`, and 200-step budget as novelty.
Next allowed test: A mechanism that repairs/constructs useful directions rather than simply rejecting most shadow steps, hard assignment-style usage constraints, Jacobian-conditioned state, or richer targets.
Source: `aiAgentWorkHistory/phase7/2026-05-28-answer-loss-step-acceptance-gate.md`
```
