# Track 3: Causal Diagnostics Scientist - Protocol, Codebooks, and Falsification

## Mission

Make it impossible for the project to fool itself with answer accuracy.

This track owns diagnostics that distinguish:

- true calculator use;
- private-code calculator use;
- ordinary transformer bypass;
- broken or irrelevant calculator paths.
- calculator-required bottleneck success versus downstream leakage or accidental bypass.

## Tactical Work

Generalize and harden the diagnostic toolkit:

- Make forced-result sweeps practical for 2-digit result vocabularies.
- Produce codebook summaries for learned operands and learned result classes.
- Add mutual-information summaries:
  - learned A vs true A;
  - learned B vs true B;
  - learned result vs true sum;
  - learned result vs first answer digit;
  - learned result vs carry / answer length.
- Add causal interventions:
  - zero calculator injection;
  - force zero result;
  - force random result;
  - force true result;
  - force learned result;
  - swap A read vectors across examples;
  - swap B read vectors across examples;
  - corrupt only A read site;
  - corrupt only B read site.
- Add bottleneck-specific falsification checks:
  - verify which downstream tensors can still carry prompt/operand information around the calculator;
  - ablate any compressed context channel independently of the calculator result;
  - force or shuffle calculator outputs in strict and relaxed bottleneck modes;
  - measure whether answer accuracy collapses when calculator result information is removed.

Prefer adding diagnostics that work on saved checkpoints so this track can proceed independently of new training runs.

## Experiments

Start by analyzing existing notable checkpoints:

- the strong 1-digit private-code Model C checkpoint;
- the weak 1-digit Model C checkpoint;
- the 2-digit learned STE collapse checkpoint;
- the 2-digit oracle checkpoints;
- Track 1's first operand-read checkpoints once available.
- any future bottleneck checkpoints, starting with oracle-bottleneck and supervised-encoder controls.

Track 2 is now available and should be treated as the first priority checkpoint set for bottleneck/leakage diagnostics. The work history is:

```text
aiAgentWorkHistory/2026-04-30-track-2-training-signal-protocol-supervision.md
```

Important Track 2 finding:

- `calculator_injection_mode=replace` replaces the residual only at active `=` positions.
- Oracle replacement works and is calculator-dependent.
- Model B/off replacement still reaches high exact match, so this is not a valid calculator-required bottleneck in the autoregressive setting; later answer-token positions can still exploit normal residual context.
- Track 3 should explicitly classify these checkpoints as invalid/leaky or partially bottlenecked before using them as evidence about calculator-required protocol learning.

Analyze these Track 2 checkpoints first:

| Purpose | Path |
| --- | --- |
| Additive answer-only learned Model C | `runs/2026-04-30_124622_062528_model-c-op0-19/model-c-2digit-seed2/final_weights.pt` |
| Additive high-answer aux `0.01` Model C | `runs/2026-04-30_124941_086322_model-c-op0-19-aux0.01/model-c-2digit-seed2/final_weights.pt` |
| Replacement Model B/off leakage control | `runs/2026-04-30_131732_611676_model-b-op0-19-replace/model-b-2digit-seed2/final_weights.pt` |
| Replacement oracle Model C control | `runs/2026-04-30_131816_053136_model-c-oracle-op0-19-replace/model-c-2digit-seed2/final_weights.pt` |
| Replacement answer-only learned Model C | `runs/2026-04-30_131959_337278_model-c-op0-19-replace/model-c-2digit-seed2/final_weights.pt` |
| Replacement aux decay best transient candidate | `runs/2026-04-30_132810_628910_model-c-op0-19-replace-aux0.03-auxdecay1000/model-c-2digit-seed2/final_weights.pt` |

Track 2 also exposed a diagnostic edge case: under `calculator_read_position=operands`, early/untrained generation may emit extra `=` tokens. Current code uses the first prompt `=` as the operand-read anchor. Track 3 diagnostics should preserve this behavior and avoid reintroducing an exactly-one-`=` assumption during generation-time analyses.

For each checkpoint, produce a compact report:

- summary metrics JSON;
- per-example CSV where useful;
- codebook CSV;
- short interpretation in `aiAgentWorkHistory`.

For bottleneck checkpoints, also produce a leakage report that states whether the run is:

- strict bottleneck: downstream sees only calculator result plus minimal formatting/position information;
- relaxed bottleneck: downstream sees calculator result plus a compressed context vector;
- invalid or leaky bottleneck: downstream can recover enough operand/task information to bypass the calculator.

## Required Metrics

At minimum:

- answer exact match;
- operand exact match;
- calculator result accuracy;
- learned result distribution;
- forced-class sweep summary;
- mutual information estimates;
- counterfactual exact-match table;
- bottleneck/leakage classification when applicable;
- example rows for successes and failures;
- exact checkpoint path.

## Success Criteria

The diagnostic suite should be able to classify any serious checkpoint into one of these categories:

- intended true-operand calculator use;
- semantically decodable private calculator code;
- causally useful but opaque private code;
- calculator ignored / bypassed;
- calculator harmful.
- valid calculator-required bottleneck use;
- invalid/leaky bottleneck where apparent success may not depend on the calculator.

This track succeeds if future experimental claims become much harder to overstate.

## Guardrails

- Do not train new models unless a diagnostic truly requires it.
- Do not infer calculator use from answer accuracy.
- Keep output formats stable so other tracks can consume the same metrics.
- Be explicit when a private code is interesting but not the intended protocol.
- For bottleneck experiments, explicitly verify that the architecture actually blocks or limits bypass before interpreting success.
