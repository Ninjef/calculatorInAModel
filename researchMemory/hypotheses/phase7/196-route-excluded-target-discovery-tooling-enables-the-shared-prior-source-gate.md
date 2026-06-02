# Route-excluded target discovery tooling enables the shared-prior source gate.

Kind: hypothesis_memory
Status: TOOLING
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-route-excluded-target-discovery-tooling.md

Summary:

- Added `--result-boundary-target-memory-update-exclude-routes`, which skips
  prompt-memory sparse target discovery on selected routed hook ids. The
  prompt-memory expected-full count is computed from score-eligible routes
  only, so freeze-when-full still works when withheld routes never receive
  direct target entries. Prior replay remains global over train/heldout prompt
  pools, so a shared numeric prior can still apply pseudo-target pressure to
  the excluded routes.

Questions:

- How do we train a routed source with target discovery disabled on some routes?
- What flag excludes routed hooks from prompt-memory target discovery?
- Has route-excluded target discovery tooling been implemented?
- Can prompt memory freeze when some routes are withheld from target discovery?
- What is the next shared-prior source gate after route-heldout diagnostics?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-route-excluded-target-discovery-tooling.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- `tests/test_model.py`

Do Not Repeat:

- Do not reimplement route-withheld prompt-memory masking. Use
  `--result-boundary-target-memory-update-exclude-routes`.
- Do not treat this tooling as proof that shared-prior source training works.

Next Allowed:

- Run the actual routed source gate: disable or reduce direct target discovery
  for one or more routes, train the shared/global numeric prior from the scored
  routes, use prior replay on all routes, then require source heldout quality
  and trusted frozen-policy additive handoff with low controls.
