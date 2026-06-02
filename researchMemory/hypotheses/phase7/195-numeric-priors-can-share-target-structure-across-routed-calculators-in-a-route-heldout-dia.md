# Numeric priors can share target structure across routed calculators in a route-heldout diagnostic.

Kind: hypothesis_memory
Status: PARTIAL-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-route-heldout-shared-prior-diagnostic.md

Summary:

- Extended `scripts/diagnose_amortized_prior_from_trace.py` with `--split-mode route_heldout` and fit h128 numeric priors on the op29 capped source trace while withholding one `left_operand_mod` route/calculator from the fit memory. Training on the other three routes reached heldout-route accuracy `0.9333`, `0.9683`, `0.9793`, and `0.9583` for routes `0-3`. The route-0 embedding-prior control fit train routes perfectly (`1.0000`) but got `0.0000` on the heldout route, showing the positive is structured numeric sharing rather than generic prompt memorization. This is not yet a full source run that skips target discovery on some calculators.

Questions:

- What did we learn about Numeric priors can share target structure across routed calculators in a route-heldout diagnostic?
- Has Numeric priors can share target structure across routed calculators in a route-heldout diagnostic been tested?
- Should we repeat Numeric priors can share target structure across routed calculators in a route-heldout diagnostic?
- What is the status of Numeric priors can share target structure across routed calculators in a route-heldout diagnostic?
- What follow-up is allowed for Numeric priors can share target structure across routed calculators in a route-heldout diagnostic?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-route-heldout-shared-prior-diagnostic.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun route-heldout diagnostics as a seed/route ladder; all four routes already passed numerically and the embedding control exposed memorization failure.

Next Allowed:

- Train a routed source where sparse target discovery is intentionally disabled or reduced for some routes and a shared/global numeric prior supplies those route targets during source training, then run the trusted handoff gate.

Full Text:

```text
PARTIAL-POSITIVE: Numeric priors can share target structure across routed calculators in a route-heldout diagnostic.
Conclusion: Extended `scripts/diagnose_amortized_prior_from_trace.py` with `--split-mode route_heldout` and fit h128 numeric priors on the op29 capped source trace while withholding one `left_operand_mod` route/calculator from the fit memory. Training on the other three routes reached heldout-route accuracy `0.9333`, `0.9683`, `0.9793`, and `0.9583` for routes `0-3`. The route-0 embedding-prior control fit train routes perfectly (`1.0000`) but got `0.0000` on the heldout route, showing the positive is structured numeric sharing rather than generic prompt memorization. This is not yet a full source run that skips target discovery on some calculators.
Do not repeat: Do not rerun route-heldout diagnostics as a seed/route ladder; all four routes already passed numerically and the embedding control exposed memorization failure.
Next allowed test: Train a routed source where sparse target discovery is intentionally disabled or reduced for some routes and a shared/global numeric prior supplies those route targets during source training, then run the trusted handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-06-02-route-heldout-shared-prior-diagnostic.md`
```
