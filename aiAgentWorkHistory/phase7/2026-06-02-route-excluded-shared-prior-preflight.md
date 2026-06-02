# 2026-06-02 - Route-excluded shared-prior preflight

## Question

Does the route-excluded target-discovery switch actually operate in the
prompt-keyed training path, and is the shared numeric-prior mechanism ready for
the full routed source/handoff gate?

## Bug Fix

The first route-exclusion commit put `memory_update_exclude_routes` on the
fixed-batch `result_boundary_online_hard_memory_loss(...)` call instead of the
prompt-keyed `result_boundary_prompt_hard_memory_loss(...)` call. The intended
shared-prior recipe uses prompt-keyed online memory, so the first op9 preflight
did not actually exclude routed prompt-memory updates.

Fix:

- Removed `memory_update_exclude_routes` from the fixed-batch online-memory
  branch.
- Passed `result_boundary_memory_update_exclude_routes` into the prompt-keyed
  memory branch.
- Added a training-loop regression asserting that the route-exclusion argument
  is threaded only to `result_boundary_prompt_hard_memory_loss(...)`.

## Verification

Focused regression after the fix:

```bash
python3 -m pytest tests/test_model.py -k "prompt_keyed_online_hard_memory or streaming_heldout_split or amortized_prior"
```

Result: `4 passed, 151 deselected`.

Syntax:

```bash
python3 -m py_compile scripts/overfit_one_batch.py
```

Result: passed.

Training-loop smoke after the fix:

```text
runs/2026-06-02_route_exclusion_training_loop_smoke/...
```

The smoke showed prompt-memory update exclusion in the actual training loop:

| Step | Expected entries | Entries | Score-eligible fraction | Update-excluded fraction | Forced evals |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `6` | `0` | `1.00` | `0.00` | `4` |
| `1` | `6` | `0` | `0.75` | `0.25` | `3` |
| `2` | `6` | `0` | `0.50` | `0.50` | `2` |

This is the evidence that route exclusion is now active in the prompt-keyed
training branch.

## Invalid Preflight

Run:

```text
runs/route_excluded_shared_prior_preflight_op9_steps800
```

Result: final exact `0.470`, train exact `0.55`, heldout exact `0.20`,
memory entries `58` versus expected `55`.

Interpretation: invalid as route-exclusion evidence because the training-loop
bug meant prompt-memory updates were not excluded. Do not use this run to judge
shared-prior source viability.

## Fixed Preflight

Run:

```text
runs/route_excluded_shared_prior_preflight_op9_steps800_fixed/2026-06-02_171114_412669_model-c-op0-9-fullgrid-streamb32-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts16-rbtuniq-rbttopk8-rbt-bac9a99900/model-c-2digit-seed9
```

Configuration:

- op0-9 four-hook `left_operand_mod` routed shared-output source.
- Streaming train batch `32`, heldout fraction `0.2`.
- Prompt-keyed online hard memory, `--result-boundary-target-memory-update-exclude-routes 1`.
- Numeric amortized prior replay with full-memory fit every `2` steps.
- `800` source steps.

Results:

| Metric | Value |
| --- | ---: |
| Final exact / calculator-result accuracy | `0.510` |
| Train prompt exact / calculator-result accuracy | `0.575` |
| Heldout prompt exact / calculator-result accuracy | `0.050` |
| Prompt-memory entries / expected | `55 / 55` |
| Forced-result evals | `7,232` |
| Prior updates | `400` |
| Prior train accuracy | `0.2909` |
| Prior heldout accuracy | `0.0500` |
| Last-row score-eligible fraction | `0.78125` |
| Last-row update-excluded fraction | `0.21875` |

Routed diagnostics:

| Hook / route | Overall calc | Train calc | Heldout calc |
| ---: | ---: | ---: | ---: |
| `0` | `0.8000` | `0.8696` | `0.0000` |
| `1` excluded | `0.0385` | `0.0800` | `0.2000` |
| `2` | `0.5417` | `0.6667` | `0.0000` |
| `3` | `0.5600` | `0.8235` | `0.0000` |

No trusted handoff was run because the source gate missed badly.

## Interpretation

The fixed preflight is a useful negative/diagnostic result, not a full source
gate. It proves the prompt-keyed route-exclusion path is now wired, but the
small op9 `800`-step shared-prior source did not train the excluded route and
the prior itself was weak. The likely blocker is that a route-excluded source
needs the stronger op19/op29 capped-prior fit dynamics or a more explicitly
global/shared prior objective; this quick preflight should not be counted as a
completed shared-prior algorithm failure.

## Next

Allowed next high-leverage tests:

- Full op19 route-excluded shared-prior source with the known strong
  numeric-prior replay dynamics, unbuffered logging, and trusted handoff only
  if the source heldout/excluded-route gate passes.
- A stronger shared/global prior objective that fits route-shared structure
  directly instead of relying on the weak small preflight setup.

Do not run more op9 short preflights, route-heldout diagnostic ladders, or
cap/seed tweaks as novelty.
