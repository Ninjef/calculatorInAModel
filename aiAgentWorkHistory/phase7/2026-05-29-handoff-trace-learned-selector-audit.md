# 2026-05-29 Handoff Trace Learned Selector Audit

## Aim

Test whether a simple learned selector trained on accumulated additive handoff
traces can replace or improve on raw 500/600-step handoff exact-match
selection.

This follows the short-slope and geometry-selector negatives. The selector
uses actual existing frozen-policy additive handoff traces as labels, not
geometry-only or 25/50/100-step loss-slope proxies.

## Script

Added:

```text
scripts/analyze_handoff_trace_selector.py
```

The script auto-discovers initial frozen-policy additive handoff runs under
the Phase 7 run roots, deduplicates repeated 600/800 variants of the same
candidate, groups by source family, and runs leave-family-out ridge regression
to predict the 600-step handoff exact-match score from earlier trace features.

Features at the chosen prediction step:

- normal exact match
- injection-zero exact match
- oracle exact match
- forced-random exact match
- calculator-result accuracy
- normal-minus-zero
- oracle-minus-normal
- normal-minus-forced-random

Baselines:

- raw early exact-match winner
- calculator-result-accuracy winner

## Runs

Run root:

```text
runs/2026-05-29_phase7_handoff_trace_selector_audit_v2
```

Commands:

```text
python3 scripts/analyze_handoff_trace_selector.py --prediction-step 200 --target-step 600 --output-root runs/2026-05-29_phase7_handoff_trace_selector_audit_v2/pred200_target600
python3 scripts/analyze_handoff_trace_selector.py --prediction-step 300 --target-step 600 --output-root runs/2026-05-29_phase7_handoff_trace_selector_audit_v2/pred300_target600
python3 scripts/analyze_handoff_trace_selector.py --prediction-step 400 --target-step 600 --output-root runs/2026-05-29_phase7_handoff_trace_selector_audit_v2/pred400_target600
python3 scripts/analyze_handoff_trace_selector.py --prediction-step 500 --target-step 600 --output-root runs/2026-05-29_phase7_handoff_trace_selector_audit_v2/pred500_target600
```

## Dataset

The audit discovered `23` trace runs, deduped them to `21` candidates, and
evaluated `8` eligible source families:

```text
src2
src4
src5
src6
src7
src9_nodecay
src10_nodecay
src10_improve5
```

## Result

Leave-family-out winner accuracy against the 600-step handoff target:

| Prediction step | Ridge selector | Raw early exact | Calc accuracy |
| ---: | ---: | ---: | ---: |
| `200` | `3/8` | `5/8` | `3/8` |
| `300` | `4/8` | `4/8` | `1/8` |
| `400` | `3/8` | `4/8` | `3/8` |
| `500` | `5/8` | `6/8` | `3/8` |

The learned ridge selector did not beat the raw early exact-match baseline at
any audited prediction step. At step `500`, where the trace is most
informative, raw early exact selected the 600-step winner in `6/8` families
while ridge selected `5/8`.

Step-500 ridge still misselected:

- `src6`: picked step `1500` instead of final;
- `src7`: picked step `1000` instead of step `1400`;
- `src10_nodecay`: picked step `1300` instead of step `1000`.

## Decision

Label:

```text
handoff_trace_ridge_selector_negative
```

A simple leave-family-out ridge model over early handoff trace features is not
a validated replacement for actual 500/600-step handoff exact-match selection.

## Interpretation

This does not mean learned selectors are impossible. It means the currently
available low-capacity trace-feature model does not justify replacing the
handoff probe. The raw handoff exact trace is still the stronger signal.

The next useful implementation is an in-source-training logging/selection
probe that periodically runs the real additive handoff probe on cloned model
state. That remains expensive, but it optimizes source checkpoint selection
against the actual behavior that has validated so far, rather than another
failed cheap proxy.

## Anti-Rerun Note

Do not repeat this same leave-family-out ridge selector over the same
prediction-step features (`200/300/400/500`) and Phase 7 handoff trace dataset
as novelty.

Next useful tests:

- implement logging-only in-training additive handoff probes for source
  checkpoints, keeping them off the gradient path;
- collect more diverse labeled handoff families before trying a higher-capacity
  learned selector;
- if trying another learned selector, validate leave-family-out against the
  same 500/600-step handoff target and require it to beat raw early exact.

## Verification

`python3 -m py_compile scripts/analyze_handoff_trace_selector.py` passed.
`git diff --check` passed. The audit wrote JSON and CSV outputs under the run
root above.
