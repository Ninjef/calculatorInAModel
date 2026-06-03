# 2026-06-03 - Candidate-Evidence Route-Excluded Source

## Purpose

Test whether direct amortized-prior updates from current-batch
candidate-scored targets can fix the op19 route-excluded shared-prior source
gate. This was the next mechanism recommended after route replay and
prior-bootstrap prompt memory both failed.

## Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_stop1pat100_exclroute1_candev1_src5000/2026-06-02_190424_817657_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-e085eed8de/model-c-2digit-seed9
```

Configuration delta from the no-bootstrap route-excluded source:

- Added `--result-boundary-target-amortized-prior-candidate-evidence-weight 1.0`.
- Kept route replay disabled.
- Kept prior-bootstrap prompt memory disabled.

## Results

- Final eval exact/calculator-result accuracy: `309/400 = 0.7725`.
- Final loss: `0.779283881187439`.
- Best diagnostic snapshot normal/calculator-result accuracy: `0.8000` at
  step `5000`.
- Final snapshot controls: injection-zero `0.0475`, forced-zero `0.0025`,
  forced-random `0.0025`.
- Diagnostic 128 exact/calculator-result accuracy: `0.796875`.
- Diagnostic routes: hook0 `0.8085`, hook1 `0.7429`, hook2 `0.7727`,
  hook3 `0.8750`.
- Train prompts exact/calculator-result accuracy: `0.80625`.
- Heldout prompts exact/calculator-result accuracy: `0.5375`.
- Train routes: hook0 `0.8763`, hook1 `0.6495`, hook2 `0.8305`,
  hook3 `0.9104`.
- Heldout routes: hook0 `0.2609`, hook1 `0.6522`, hook2 `0.6190`,
  hook3 `0.6923`.
- Prompt memory entries/expected direct entries: `223/223`.
- Forced-result evals: `33,816`.
- Prior updates: `2,501`.
- Candidate-evidence prior updates/examples: `32/1060`.
- Prior train/heldout accuracy: `0.7156/0.5375`.
- Prior train/heldout confidence: `0.3270/0.3014`.

Candidate-evidence timing:

- Prompt memory reached `223` direct entries by step `50`.
- Logged nonzero candidate-evidence rows appeared only at steps `0` and `25`.
- Step `0`: `33` candidate-evidence targets, target-vs-true accuracy `0.8788`,
  confidence `0.0382`, prior-vs-target accuracy `0.0606`.
- Step `25`: `28` candidate-evidence targets in the logged batch, cumulative
  `26` updates and `849` examples, target-vs-true accuracy `0.8929`,
  confidence `0.0454`, prior-vs-target accuracy `0.2143`.

## Interpretation

Mixed-negative. The candidate-evidence update path fired and reused already
scored candidates, but it did not improve the live route-excluded source gate.
Final quality is below the no-bootstrap source and below the route-replay
variant; heldout prompts are worse than all prior full op19 route-excluded
variants.

No trusted handoff was run because heldout source quality and excluded-route
quality missed the gate.

## Next

Close the route-excluded tweak branch. Do not run candidate-evidence weight or
timing ladders as novelty. Move to shared/global target formation, joint
target learning across routes, or a less-prescriptive credit signal that
removes per-route prompt-memory tables and answer-derived candidate scoring.
