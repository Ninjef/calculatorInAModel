# Hard-path answer-loss line search over proposed shadow step scales rescues refreshed online-shadow Stage 1.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-answer-loss-line-search-gate.md

Summary:

- Scales `1,0.5,0.25,0.1,0` accepted only `5/200` steps (`2.5%`); best snapshot improved to `0.0925`, but final exact was only `0.060`.

Questions:

- What did we learn about Hard-path answer-loss line search over proposed shadow step scales rescues refreshed online-shadow Stage 1?
- Has Hard-path answer-loss line search over proposed shadow step scales rescues refreshed online-shadow Stage 1 been tested?
- Should we repeat Hard-path answer-loss line search over proposed shadow step scales rescues refreshed online-shadow Stage 1?
- What is the status of Hard-path answer-loss line search over proposed shadow step scales rescues refreshed online-shadow Stage 1?
- Why did Hard-path answer-loss line search over proposed shadow step scales rescues refreshed online-shadow Stage 1 fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-answer-loss-line-search-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same refreshed h32 validation-gradient module with feedback clamp `10`, answer-loss line-search scales `1,0.5,0.25,0.1,0`, and 200-step budget as novelty.

Next Allowed:

- Construct better directions, hard assignment-style usage constraints, Jacobian-conditioned state, or richer targets rather than selecting among mostly harmful proposed shadow steps.

Full Text:

```text
DISPROVEN: Hard-path answer-loss line search over proposed shadow step scales rescues refreshed online-shadow Stage 1.
Conclusion: Scales `1,0.5,0.25,0.1,0` accepted only `5/200` steps (`2.5%`); best snapshot improved to `0.0925`, but final exact was only `0.060`.
Do not repeat: Same refreshed h32 validation-gradient module with feedback clamp `10`, answer-loss line-search scales `1,0.5,0.25,0.1,0`, and 200-step budget as novelty.
Next allowed test: Construct better directions, hard assignment-style usage constraints, Jacobian-conditioned state, or richer targets rather than selecting among mostly harmful proposed shadow steps.
Source: `aiAgentWorkHistory/phase7/2026-05-28-answer-loss-line-search-gate.md`
```
