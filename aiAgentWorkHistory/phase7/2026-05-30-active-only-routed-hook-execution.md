# 2026-05-30 - Active-only routed hook execution

Task: reduce the many-calculator compute waste exposed by the four-hook routed
source/handoff result.

## Question

Can routed multi-hook execution avoid calling every calculator hook on every
batch?

The previous four-hook result proved that routed hooks can train and transfer,
but the implementation still ran every hook on the full batch and masked
injections afterward. That was not a scalable execution story.

## Changes

- Updated `TinyGPT._call_calculator_hooks` so routed mode computes route masks
  first, invokes only hooks with examples in the batch, and scatters each
  active hook's injection/trace back into full-batch diagnostics.
- Kept per-hook full-batch injection/trace buffers diagnostic-only so ordinary
  training does not pay for routed observability artifacts.
- Added `calculator_invoked_hook_count` while preserving
  `calculator_active_hook_count` as the configured hook count.
- Updated routed source-training result-logit reads so each present hook's
  `result_proj` is applied only to examples routed to that hook, instead of
  stacking every hook's logits over the full batch.
- Added routed summary fields for configured and invoked hook counts.
- Added regression tests for active-only forward calls and active-only routed
  result projections.

## Validation

Focused routed tests:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py::test_multiple_calculator_hooks_sum_independent_injections tests/test_model.py::test_left_operand_mod_routes_examples_to_one_hook tests/test_model.py::test_routed_calculator_only_invokes_present_hooks tests/test_model.py::test_routed_result_policy_reads_active_hook_logits tests/test_model.py::test_routed_result_policy_only_projects_present_hooks -q
```

Result: `5 passed`.

Broader regression set:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py tests/test_assignment_scaling.py tests/test_research_memory.py -q
```

Result: `146 passed`.

## Result

This is a positive implementation result for many-calculator scaling. Routed
batches no longer pay the forward cost for hooks with no examples, and the
source-training result-policy read path no longer applies every routed hook's
result projection to every example.

## Interpretation

Active-only routed execution fixes a real scalability flaw in the routed
architecture, but it is not a new credit-assignment method and not a complete
many-calculator solution. The current positive routed recipe still uses cloned
or independent output projections, so parameter scaling remains unresolved.

Next high-leverage step: implement shared/tied output projections or explicit
compute accounting in a routed training gate, then return to the larger
non-prescriptive credit-assignment problem.
