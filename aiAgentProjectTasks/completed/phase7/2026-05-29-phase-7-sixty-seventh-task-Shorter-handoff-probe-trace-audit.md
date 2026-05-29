# Phase 7 Sixty-Seventh Task: Shorter Handoff Probe Trace Audit

## Status

Completed 2026-05-29.

## Question

Can the source-selection handoff probe be shortened below 600 steps without
changing the selected source checkpoint on the current validated candidate
sets?

## Setup

- Reused existing frozen-policy additive handoff probe traces.
- Audited `src5` candidates step `1100`, step `1400`, step `1500`, and final.
- Audited `src4` candidates step `1000`, step `1200`, and final.
- Compared normal accuracy at probe steps `400`, `500`, and `600`.

## Result

| Source family | 400-step selector | 500-step selector | 600-step selector |
| --- | --- | --- | --- |
| `src5` | step `1500` | step `1100` | step `1100` |
| `src4` | step `1200` | step `1200` | step `1200` |

## Decision

```text
shorter_handoff_probe_500_trace_positive_400_negative
```

The 400-step probe is too short for `src5`, but 500 steps selects the same
checkpoint as 600 steps on both current validated source families.

## Next

Validate the 500-step selector on newly acquired source checkpoints, reduce the
800-step continuation stage, or train source policies directly for early
handoff and continuation slope.
