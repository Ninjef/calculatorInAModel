# Canonical Calculator Diagnostics

Phase-2 work should use phase-neutral diagnostic names in new reports and task
notes. The older Track 3 and Track 4 labels are still useful historical shorthand,
and the legacy script paths remain backward-compatible wrappers.

## Causal Calculator Protocol Diagnostics

Canonical entrypoints:

```bash
python3 scripts/diagnose_calculator_protocol.py --checkpoint <run>/final_weights.pt ...
python3 scripts/run_causal_calculator_protocol_diagnostics.py --checkpoint <run>/final_weights.pt ...
```

Legacy manifest runner:

```bash
python3 scripts/run_phase1_track3_causal_diagnostics.py ...
```

This diagnostic is for causal dependence and bottleneck classification. It
reports learned calculator actions, calculator result accuracy, exact match,
counterfactual answer accuracy, protocol/codebook summaries, and labels such as
`calculator_ignored_or_bypassed`, `causally_useful_opaque_private_code`, and the
bottleneck classification. With `--forced-result-sweep`, it also performs the
forced-result sweep to measure whether the downstream decoder prefers the true
result class, the learned result class, or another forced class.

## Action-Loss Diagnostics

Canonical entrypoint:

```bash
PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py --checkpoint <run>/final_weights.pt ...
```

Legacy wrappers:

```bash
PYTHONPATH=. python3 scripts/run_track4_action_loss_diagnostic.py ...
PYTHONPATH=. python3 scripts/run_phase1_track4_action_loss_diagnostic.py ...
```

This diagnostic is for learned-vs-true-vs-random/shuffled action NLL comparison.
It asks whether the downstream answer loss actually rewards better calculator
operand actions for a checkpoint. The key gaps are learned-minus-true,
random-minus-true, and shuffled-minus-true NLL, plus the true-best and
learned-best fractions.

## What These Are Not

These diagnostics are not unit tests. They are checkpoint-level research probes
with sampled prompts and intervention settings.

They are not phase-1-only task files. The Track 3 and Track 4 labels describe
where the tools came from historically, not a limit on where they apply.

They are not proof of success by themselves. A successful adaptive-interface
checkpoint still needs calculator-dependent answer accuracy and learned
calculator-action improvement under the intended bottleneck. In particular,
oracle-at-eval recovery or a strong true-action loss gap only shows the frozen
downstream decoder has useful signal; it does not show the learned interface has
found the action protocol.
