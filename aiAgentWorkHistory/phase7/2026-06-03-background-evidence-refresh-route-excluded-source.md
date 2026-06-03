# 2026-06-03 Background Evidence Refresh Route-Excluded Source

## Question

Can background candidate-evidence refresh train the shared numeric prior well
enough to cover a route excluded from direct prompt-memory target discovery?

## Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_stop1pat100_exclroute1_evref1_b32_e10_src5000/2026-06-02_194056_045602_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-3cc8bf63d3/model-c-2digit-seed9
```

Key settings:

- op19 four-hook shared-output source, `left_operand_mod` routing.
- Route `1` excluded from prompt-memory target discovery and background
  evidence-refresh scoring.
- Prompt-keyed online hard memory with freeze-when-full.
- Numeric amortized prior replay every `2` steps, stop train accuracy `1.0`,
  patience `100`.
- Background evidence refresh: weight `1.0`, batch `32`, every `10` steps.

## Result

Mixed-negative.

- Final eval exact/calculator-result accuracy: `252/400 = 0.6300`.
- Best/final snapshot normal: `0.6800`/`0.6475`.
- Final snapshot controls: injection-zero `0.0475`, forced-zero `0.0025`,
  forced-random `0.0025`.
- Train prompt exact/calc: `0.684375`.
- Heldout prompt exact/calc: `0.3625`.
- Prior train/heldout accuracy: `0.50625`/`0.3875`.
- Prior train/heldout confidence: `0.2332`/`0.2203`.
- Excluded route 1 train/heldout/diagnostic: `0.3505`/`0.5217`/`0.4857`.
- Prompt memory entries/expected direct entries: `223/223`.
- Online prompt-memory forced evals: `42,144`.
- Prior updates: `2,501`.
- Evidence-refresh updates/examples/forced evals: `501`/`11,056`/`267,216`.

No trusted additive handoff was run because source heldout and excluded-route
quality missed.

## Interpretation

Background refresh fired heavily and respected the route-exclusion setup, but
it did not improve the op19 route-excluded source gate. It degraded final
source accuracy, prompt heldout accuracy, prior heldout accuracy, and excluded
route quality relative to the no-refresh and candidate-evidence sources.

This closes the current route-excluded tweak branch. Do not run refresh
batch/every/weight/exclude-route ladders as novelty. The next work should move
to genuinely shared/global target formation, joint target learning across
routes, or less-prescriptive credit that removes per-route prompt-memory target
tables and answer-derived candidate scoring.

## Verification

- Source run completed successfully.
- Metrics were read from `metrics.json`, `diagnostic_snapshots.csv`, and routed
  diagnostic summaries in the run directory.
