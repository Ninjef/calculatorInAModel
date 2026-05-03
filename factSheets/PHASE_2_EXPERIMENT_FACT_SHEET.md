# Phase 2 Experiment Fact Sheet

## Adaptive interface under strict answer-decoder bottleneck

- Date: 2026-05-01.
- Task: `aiAgentProjectTasks/2026-05-01-phase-2-first-task-Adaptive-calculator-interface-bottleneck.md`.
- Work history: `aiAgentWorkHistory/phase2/2026-05-01-adaptive-calculator-interface-bottleneck.md`.
- Starting decoder checkpoint: `runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt`.
- Adaptive run: `runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2`.
- Config: `digits=2`, `operand_max=19`, operand vocab `20`, `2L/1H/16d/mlp1`, hook after layer `1`, `calculator_read_position=operands`, `calculator_bottleneck_mode=answer_decoder`, `calculator_estimator=adaptive_interface`.
- Frozen by default: `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder`.
- Upstream encoder trainable: yes.
- `calculator_hook.input_proj` moved substantially from the oracle checkpoint: weight L2 delta `13.1145`, bias L2 delta `2.6793`.
- Frozen decoder components stayed unchanged: output projection, answer offset embedding, and answer decoder all had L2 delta `0.0000`.

Headline result:

- Eval exact match: `5/512 = 0.0098`.
- Built-in 128-sample diagnostic exact match: `0.0156`.
- Operand exact match: `0.0000`.
- Calculator result accuracy: `0.0156`.
- Adaptive target result accuracy: `0.9375`.
- Learned-target result agreement: `0.0156`.
- Adaptive target operand exact match: `0.1172`.
- Final adaptive interface CE loss: `193.0297`.

Counterfactuals:

- Injection-zero exact: `0.0078`.
- Forced-zero exact: `0.0000`.
- Forced-random exact: `0.0156`.
- Oracle-at-eval exact: `0.9531`.

Canonical causal protocol diagnostic (legacy Track 3):

- Output: `runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2/track3_diagnostics`.
- Classification: `calculator_ignored_or_bypassed`.
- Bottleneck label: `strict_bottleneck_unvalidated` for this checkpoint.
- Learned calculator result collapsed to class `5` for all 64 causal diagnostic prompts.
- True-sum class was best forced class on `0.9219` of prompts.
- Learned class was best forced class on `0.0156` of prompts.

Canonical action-loss diagnostic (legacy Track 4):

- Output: `runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2/track4_action_loss`.
- Mean true-action NLL: `0.1181`.
- Mean random-action NLL: `9.9763`.
- Mean shuffled-action NLL: `10.4317`.
- Mean learned-forced NLL: `14.4446`.
- Random minus true gap: `9.8583`.
- Shuffled minus true gap: `10.3136`.
- Learned minus true gap: `14.3265`.
- True best fraction: `0.9844`.
- Learned best fraction: `0.0000`.

Conclusion:

- Useful negative result. The strict oracle decoder and action-loss landscape remain strong: true calculator actions are clearly preferred and oracle-at-eval recovers high answer accuracy.
- The adaptive moving interface did not learn that protocol. It changed substantially but collapsed to an overconfident constant calculator result, with chance answer accuracy and zero operand exact match.
- Recommendation: no-go on this exact adaptive-interface variant. Next variants should stabilize interface optimization before scaling, for example lower LR/loss weight, entropy regularization, target smoothing, upstream-frozen diagnostic training, or a small true-operand aux stabilizer while preserving the strict answer-decoder gates.

## Stabilized adaptive-interface objective ladder

- Date: 2026-05-01.
- Task: `aiAgentProjectTasks/completed/phase2/2026-05-01-phase-2-second-task-Stabilize-adaptive-interface-objective.md`.
- Work history: `aiAgentWorkHistory/phase2/2026-05-01-stabilize-adaptive-interface-objective.md`.
- Starting decoder checkpoint: `runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt`.
- Shared config: `digits=2`, `operand_max=19`, operand vocab `20`, `2L/1H/16d/mlp1`, hook after layer `1`, `calculator_read_position=operands`, `calculator_bottleneck_mode=answer_decoder`, `calculator_estimator=adaptive_interface`.
- Code changes: separate adaptive optimizer LRs (`--input-proj-lr`, `--upstream-lr`), soft result-mass objective (`--adaptive-interface-target-mode soft_result`), and adaptive entropy bonus (`--adaptive-interface-entropy-weight`).
- Frozen semantic components stayed frozen in every run: `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder` all had L2 delta `0.0000`.

Run comparison:

| Variant | Run | Eval exact | Operand exact | Calc result acc | Target acc | Learned-target agreement | Final CE | Causal diagnostic | Action-loss learned-true gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Frozen upstream | `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2` | `0.1016` | `0.0234` | `0.1016` | `0.9375` | `0.0625` | `2.8654` | `causally_useful_opaque_private_code` | `9.0455` |
| Lower LR | `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` | `0.0664` | `0.0156` | `0.0469` | `0.9375` | `0.0703` | `2.2887` | `causally_useful_opaque_private_code` | `10.4115` |
| Soft result | `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-soft_result-answer_decoder/model-c-2digit-seed2` | `0.0059` | `0.0000` | `0.0156` | `0.9375` | `0.0078` | `226.3104` | `calculator_ignored_or_bypassed` | `8.4456` |
| Lower LR + entropy `0.001` | `runs/2026-05-01_081215_279041_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-ient0.001-answer_decoder/model-c-2digit-seed2` | `0.0410` | `0.0078` | `0.0625` | `0.9375` | `0.0625` | `2.3270` | `causally_useful_opaque_private_code` | `10.2226` |
| Lower LR + entropy `0.003` | `runs/2026-05-01_081215_279020_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-ient0.003-answer_decoder/model-c-2digit-seed2` | `0.0254` | `0.0000` | `0.0469` | `0.9375` | `0.0234` | `2.3133` | `calculator_ignored_or_bypassed` | `9.2909` |
| Lower LR + entropy `0.01` | `runs/2026-05-01_081215_283303_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-ient0.01-answer_decoder/model-c-2digit-seed2` | `0.0391` | `0.0000` | `0.0469` | `0.9375` | `0.0859` | `2.3345` | `calculator_ignored_or_bypassed` | `6.9002` |

Counterfactual and action-loss notes:

- Built-in counterfactuals were identical across the six runs: injection-zero `0.0078`, forced-zero `0.0000`, forced-random `0.0156`, oracle-at-eval `0.9531`.
- Canonical causal diagnostic oracle-at-eval was `0.90625` for every run; true-sum forced result class was best on `0.921875` of prompts for every run.
- Canonical action-loss diagnostic true-action NLL stayed `0.1181`, random-action NLL `9.9763`, shuffled-action NLL `10.4317`; learned actions were never best (`learned_best_fraction=0.0`) in any run.

Conclusion:

- Useful negative result. Frozen-upstream training improved normal/calculator-result accuracy over the failed baseline, so encoder/interface co-adaptation is likely part of the instability.
- Lower LR reduced `input_proj` movement and held the adaptive CE down, but still learned a constant calculator result under the canonical causal diagnostic.
- Soft-result target alone did not remove collapse; it became overconfident and worse than the hard-pair objective.
- Entropy helped keep distributions less sharp but did not produce stable learned calculator actions. High entropy is not success: operand exact remained near zero and action-loss learned-best stayed `0.0`.
- Recommendation: no-go on adaptive-interface success. Next variants should keep the strict decoder and consider staged/slow interface training, target smoothing or clipping, or a small true-operand auxiliary stabilizer as a diagnostic.

## Canonical diagnostics and true-operand stabilizer ladder

- Date: 2026-05-01.
- Task: `aiAgentProjectTasks/2026-05-01-phase-2-third-task-Canonical-diagnostics-and-staged-interface-stabilizer.md`.
- Work history: `aiAgentWorkHistory/phase2/2026-05-01-canonical-diagnostics-staged-interface-stabilizer.md`.
- Canonical diagnostic contract: `docs/canonical_diagnostics.md`.
- Canonical causal protocol entrypoints: `scripts/diagnose_calculator_protocol.py` and `scripts/run_causal_calculator_protocol_diagnostics.py`.
- Canonical action-loss entrypoint: `scripts/run_action_loss_diagnostic.py`.
- Backward-compatible legacy wrappers remain available for phase-1 Track 3/Track 4 paths.
- Starting decoder checkpoint: `runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt`.
- Shared training config: `digits=2`, `operand_max=19`, operand vocab `20`, `2L/1H/16d/mlp1`, hook after layer `1`, `calculator_read_position=operands`, `calculator_bottleneck_mode=answer_decoder`, `calculator_estimator=adaptive_interface`, `freeze_semantic_decoder=true`, `input_proj_lr=3e-4`, `upstream_lr=3e-4`, `adaptive_interface_target_mode=hard_pair`.
- All stabilizer runs decayed the true-operand aux weight to `0.0` before final evaluation.
- Frozen semantic components stayed frozen: `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder` all had L2 delta `0.0000`.

Run comparison:

| Variant | Run | Eval exact | Diag exact | Operand exact | Calc result acc | Target acc | Learned-target agreement | Final CE | Aux loss / weight | Causal diagnostic | Action-loss learned-true gap | Learned best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Failed adaptive baseline | `runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2` | `0.0098` | `0.0156` | `0.0000` | `0.0156` | `0.9375` | `0.0156` | `193.0297` | `0.0000 / 0.0000` | `calculator_ignored_or_bypassed` | `14.3265` | `0.0000` |
| Lower LR baseline | `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` | `0.0664` | `0.0469` | `0.0156` | `0.0469` | `0.9375` | `0.0703` | `2.2887` | `0.0000 / 0.0000` | `causally_useful_opaque_private_code` | `10.4115` | `0.0000` |
| Aux `0.01`, decay `250` | `runs/2026-05-01_083538_751519_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.01-auxdecay250/model-c-2digit-seed2` | `0.0078` | `0.0156` | `0.0078` | `0.0156` | `0.9375` | `0.0000` | `2.3514` | `5.7272 / 0.0000` | `calculator_ignored_or_bypassed` | `8.7436` | `0.0000` |
| Aux `0.01`, decay `500` | `runs/2026-05-01_084030_058179_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.01-auxdecay500/model-c-2digit-seed2` | `0.0176` | `0.0312` | `0.0000` | `0.0312` | `0.9375` | `0.0312` | `2.3530` | `5.7951 / 0.0000` | `calculator_ignored_or_bypassed` | `10.3902` | `0.0000` |
| Aux `0.03`, decay `250` | `runs/2026-05-01_084523_733373_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.03-auxdecay250/model-c-2digit-seed2` | `0.0156` | `0.0156` | `0.0078` | `0.0156` | `0.9375` | `0.0000` | `2.3501` | `6.0152 / 0.0000` | `calculator_harmful` | `10.5551` | `0.0000` |
| Aux `0.03`, decay `500` | `runs/2026-05-01_085013_079390_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.03-auxdecay500/model-c-2digit-seed2` | `0.0391` | `0.0469` | `0.0156` | `0.0469` | `0.9375` | `0.0547` | `2.2918` | `5.7369 / 0.0000` | `calculator_ignored_or_bypassed` | `9.8871` | `0.0000` |

Counterfactual and diagnostic notes:

- Built-in 64-sample canonical causal counterfactuals were identical for every aux run: injection-zero `0.0000`, forced-zero `0.0156`, forced-random `0.0625`, oracle-at-eval `0.90625`.
- The forced-result sweep continued to show a healthy downstream decoder: true-sum forced class was best on `0.921875` of prompts in every aux run.
- Canonical action-loss diagnostics continued to show true actions are available to the downstream decoder: true-action NLL `0.1181`, random-action NLL `9.9763`, shuffled-action NLL `10.4317`.
- Learned actions were never best (`learned_best_fraction=0.0`) in any aux run.
- The best aux run by learned-true action-loss gap was aux `0.01` decay `250` (`8.7436`), but it was worse than the lower-LR baseline on answer accuracy, calculator result accuracy, and learned-target agreement.
- The best aux run by answer/calculator-result accuracy was aux `0.03` decay `500`, matching the lower-LR baseline's diagnostic exact and calculator-result accuracy (`0.0469`) but not improving over it.

Conclusion:

- Useful negative result. The small decaying true-operand stabilizer did not bootstrap a stable learned calculator interface after the aux weight reached zero.
- The strict semantic decoder remains healthy, and the canonical diagnostics still show that true calculator actions would solve the task, but the learned interface does not maintain those actions.
- Recommendation: no-go on claiming adaptive-interface success. Further work should either change the interface optimization more substantially or test a deliberately supervised warm-start protocol while keeping success evidence limited to post-supervision calculator-dependent learned actions.

## Warm-started interface retention under strict bottleneck

- Date: 2026-05-01.
- Task: `aiAgentProjectTasks/2026-05-01-phase-2-fourth-task-Warm-started-interface-retention.md`.
- Work history: `aiAgentWorkHistory/phase2/2026-05-01-warm-started-interface-retention.md`.
- Code changes: added `--answer-loss-weight`; recorded trainable parameter groups, final aux CE/weight, and aux gradient-routing metadata in training outputs.
- Starting decoder checkpoint: `runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt`.
- Shared config: `digits=2`, `operand_max=19`, operand vocab `20`, `2L/1H/16d/mlp1`, hook after layer `1`, `calculator_read_position=operands`, `calculator_bottleneck_mode=answer_decoder`, `calculator_estimator=adaptive_interface`, `freeze_semantic_decoder=true`.
- Retained primary stages froze upstream and trained only `calculator_hook.input_proj`.
- Failed setup attempts showed mixed answer+aux Stage A did not warm start the interface. The successful Stage A used `answer_loss_weight=0.0`, aux weight `1.0`, adaptive weight `0.0`.

Primary staged checkpoints:

| Stage | Run | Aux weight at end | Eval exact | Trace operand exact | Trace calc result acc | Learned-target agree | Action learned-true gap | Learned best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A warm start | `runs/2026-05-01_112203_955074_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1/model-c-2digit-seed2` | `1.0` | `0.3809` | `0.2500` | `0.3750` | `0.2969` | `4.1302` | `0.0000` |
| B supervision-zero | `runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2` | `0.0` | `0.4258` | `0.4844` | `0.5078` | `0.3828` | `3.0642` | `0.0000` |
| C adaptive-only final | `runs/2026-05-01_112843_524620_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder/model-c-2digit-seed2` | `0.0` | `0.2734` | `0.2578` | `0.3516` | `0.1953` | `5.5241` | `0.0000` |

Canonical diagnostic notes:

- Stage C final evaluation happened after true-operand supervision was exactly `0.0`.
- Stage C remained calculator-dependent: canonical normal `0.2500`, injection-zero `0.0000`, forced-zero `0.0156`, forced-random `0.0625`, oracle-at-eval `0.9063`.
- Canonical causal classification for A/B/C remained `causally_useful_opaque_private_code` with `strict_bottleneck_unvalidated`.
- Forced-result sweep true-sum-best fraction stayed `0.9219`; learned-best fraction was `0.2656` at A, `0.3750` at B, and `0.2500` at C.
- Canonical action-loss true/random/shuffled NLLs stayed `0.1181`, `9.9763`, and `10.4317`; learned NLL improved versus prior baselines but was still never best.
- Frozen semantic components stayed unchanged at every primary checkpoint: `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder` deltas were all `0.0`.

Comparison to prior baselines:

- Failed adaptive baseline: eval `0.0098`, operand exact `0.0000`, calc result acc `0.0156`, learned-target agree `0.0156`, learned-true gap `14.3265`.
- Lower-LR baseline: eval `0.0664`, operand exact `0.0156`, calc result acc `0.0469`, learned-target agree `0.0703`, learned-true gap `10.4115`.
- Best previous aux stabilizer: eval `0.0391`, operand exact `0.0156`, calc result acc `0.0469`, learned-target agree `0.0547`, learned-true gap `9.8871`.
- Stage C final is materially above all three on eval exact, operand exact, calculator-result accuracy, learned-target agreement, and learned-minus-true action-loss gap, but learned-best fraction remains `0.0`.

Conclusion:

- Useful partial positive. A deliberately warm-started `input_proj` interface can retain a materially better calculator-dependent protocol after direct operand supervision is removed.
- Not a solved adaptive-interface result. Stage C degraded from the supervision-zero handoff and canonical diagnostics still label the learned protocol as opaque/private rather than a stable true-operand protocol.
- Recommendation: continue retention work with lower adaptive-only LR or slower Stage C, then test upstream unfreezing only after input-proj-only retention is stable.

## Post-supervision retention stabilization under strict bottleneck

- Date: 2026-05-01.
- Task: `aiAgentProjectTasks/2026-05-01-phase-2-fifth-task-Post-supervision-retention-stabilization.md`.
- Work history: `aiAgentWorkHistory/phase2/2026-05-01-post-supervision-retention-stabilization.md`.
- Code changes: added optional checkpoint-relative `calculator_hook.input_proj` anchor flags (`--input-proj-anchor-checkpoint`, `--input-proj-anchor-weight`, `--input-proj-anchor-decay-steps`) with training-curve and metrics logging. Anchor runs were not needed because lower-LR pure Stage C stabilized the handoff.
- Starting Stage B handoff: `runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt`.
- Shared Stage C config: `digits=2`, `operand_max=19`, operand vocab `20`, `2L/1H/16d/mlp1`, hook after layer `1`, `calculator_read_position=operands`, `calculator_bottleneck_mode=answer_decoder`, `calculator_estimator=adaptive_interface`, `freeze_semantic_decoder=true`, `freeze_upstream_encoder=true`, trainable `calculator_hook.input_proj` only.
- All Stage C variants ended with `final_aux_operand_loss_weight=0.0`.

Stage C run comparison:

| Variant | Run | Steps | Input LR | Adaptive weight | Eval exact | Trace operand exact | Trace calc result acc | Learned-target agree | Action learned-true gap | Learned best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage B target | `runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2` | 500 | `0.003` | `1.0` | `0.4258` | `0.4844` | `0.5078` | `0.3828` | `3.0642` | `0.0000` |
| Prior/control Stage C drift | `runs/2026-05-01_114850_422866_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder/model-c-2digit-seed2` | 500 | `0.003` | `1.0` | `0.2734` | `0.2578` | `0.3516` | `0.1953` | `5.5241` | `0.0000` |
| C-low-lr | `runs/2026-05-01_114850_238336_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` | 1000 | `0.0003` | `1.0` | `0.4492` | `0.5547` | `0.5859` | `0.4766` | `3.0466` | `0.0000` |
| C-very-low-lr | `runs/2026-05-01_114849_026560_model-c-op0-19-adaptive_interface-inlr0.0001-uplr0.0001-answer_decoder/model-c-2digit-seed2` | 1000 | `0.0001` | `1.0` | `0.4199` | `0.4531` | `0.4766` | `0.3672` | n/a | n/a |
| C-adaptive-low-weight | `runs/2026-05-01_114850_842904_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` | 1000 | `0.0003` | `0.3` | `0.4395` | `0.5781` | `0.5859` | `0.4609` | `3.3089` | `0.0000` |

Canonical diagnostic notes:

- `C-low-lr` canonical causal: normal `0.4375`, injection-zero `0.0000`, forced-zero `0.0156`, forced-random `0.0625`, oracle-at-eval `0.9063`, operand exact `0.4219`, calc result acc `0.4844`.
- `C-low-lr` forced-result sweep: learned-best `0.4375`, true-sum-best `0.9219`.
- `C-low-lr` action-loss: true NLL `0.1181`, learned NLL `3.1647`, random NLL `9.9763`, shuffled NLL `10.4317`, learned-true gap `3.0466`.
- `C-adaptive-low-weight` had slightly better canonical operand exact (`0.4688`) but worse learned-true action gap (`3.3089`) than `C-low-lr`.
- All diagnosed checkpoints remained `causally_useful_opaque_private_code` with `strict_bottleneck_unvalidated`.

Parameter movement:

| Checkpoint | vs Stage B weight L2/max | vs Stage B bias L2/max |
| --- | --- | --- |
| C-control-repeat | `13.6511 / 1.1142` | `4.4675 / 0.8399` |
| C-low-lr | `3.1749 / 0.2437` | `0.9003 / 0.2122` |
| C-very-low-lr | `1.1427 / 0.0839` | `0.3053 / 0.0795` |
| C-adaptive-low-weight | `3.1840 / 0.2409` | `0.8723 / 0.2112` |

Conclusion:

- Strong positive for input-proj-only Stage C retention: `input_proj_lr=0.0003`, adaptive weight `1.0`, 1000 steps, no anchors, and true-operand aux weight exactly `0.0` preserved/improved the Stage B handoff.
- Lower LR slows enough drift to keep the useful interface; this is not merely an answer-decoder artifact because injection-zero remains `0.0` and oracle-at-eval remains high.
- Remaining limitation: the interface is still canonical opaque/private-code and learned actions are never action-loss best.
- Recommendation: replicate `C-low-lr` before upstream unfreezing. Do not use anchor stabilization unless lower-LR replication regresses.

## Lower-LR retention replication and private-protocol decoding

- Date: 2026-05-01.
- Task: `aiAgentProjectTasks/2026-05-01-phase-2-sixth-task-Lower-LR-retention-replication-and-protocol-decoding.md`.
- Work history: `aiAgentWorkHistory/phase2/2026-05-01-lower-lr-retention-replication-private-protocol.md`.
- Code changes: added checkpoint-first `scripts/diagnose_private_protocol.py` for all-pair confusion matrices, per-operand/group summaries, affine/majority mappings, read-vector intervention summaries, and example rows.
- Starting Stage B handoff: `runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt`.
- Shared primary config: strict answer-decoder bottleneck, `calculator_estimator=adaptive_interface`, `adaptive_interface_target_mode=hard_pair`, `freeze_semantic_decoder=true`, `freeze_upstream_encoder=true`, `trainable_parameter_groups=[calculator_hook.input_proj]`, `answer_loss_weight=1.0`, `aux_operand_loss_weight=0.0`, no anchor.
- All primary replications and the long continuation ended with `final_aux_operand_loss_weight=0.0`.

Primary replication result:

| Checkpoint | CLI seed -> run seed | Eval exact | Canonical operand exact | Canonical calc acc | Learned-target agree | Action learned-true gap | Learned best | Classification |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Drift control | `0 -> 2` | `0.2734` | `0.2031` | `0.2656` | `0.1953` | `5.5241` | `0.0000` | `causally_useful_opaque_private_code` |
| Prior C-low-lr | `0 -> 2` | `0.4492` | `0.4219` | `0.4844` | `0.4766` | `3.0466` | `0.0000` | `causally_useful_opaque_private_code` |
| Rep seed1 | `1 -> 3` | `0.5195` | `0.4219` | `0.4688` | `0.5078` | `2.4786` | `0.0000` | `causally_useful_opaque_private_code` |
| Rep seed2 | `2 -> 4` | `0.4902` | `0.4375` | `0.4375` | `0.4688` | `2.9001` | `0.0000` | `causally_useful_opaque_private_code` |
| Rep seed3 | `3 -> 5` | `0.4746` | `0.3438` | `0.4063` | `0.4688` | `2.6357` | `0.0000` | `causally_useful_opaque_private_code` |
| Long 3000 | `0 -> 2` | `0.3613` | `0.3125` | `0.3594` | `0.3828` | `4.4470` | `0.0000` | `causally_useful_opaque_private_code` |

Canonical causal controls:

- Every diagnosed checkpoint kept injection-zero at `0.0000`, forced-zero at `0.0156`, forced-random at `0.0625`, oracle-at-eval at `0.9063`, and true-sum forced-result best at `0.9219`.
- The canonical bottleneck label stayed `strict_bottleneck_unvalidated` throughout.
- The 1000-step replications beat the high-LR drift control on the required retention metrics, but the 3000-step continuation regressed, so lower LR gives a useful retention window rather than a monotonic improvement path.

Private-protocol decoding:

| Checkpoint | All-pair answer | Operand exact | Calc result acc | Majority-mapped operand | Majority-mapped calc | Result-code majority acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage B handoff | `0.4500` | `0.4575` | `0.4875` | `0.4775` | `0.4850` | `0.4975` |
| Prior C-low-lr | `0.4875` | `0.4900` | `0.5250` | `0.4900` | `0.5250` | `0.5300` |
| Best replication | `0.5000` | `0.5100` | `0.5425` | `0.5200` | `0.5575` | `0.5475` |

Private-protocol interpretation:

- The interface is partly true-operand-like, not a clean learned permutation. The best affine modulo-20 mapping is identity for A and B; A is strong (`0.85` exact), while B remains noisy (`0.54` to `0.5975` exact).
- Majority mapping provides only small gains overall, although it helps small/no-carry examples more.
- Errors concentrate on carry and large-operand cases. For the best replication, all-pair exact is `0.5000`, carry answer exact is `0.4812`, no-carry answer exact is `0.6182`, large-operand answer exact is `0.4800`, and small-operand answer exact is `0.5600`.
- Corrupting read vectors collapses best-replication answer exact to `0.0350`/`0.0475`; swap interventions were no-ops in this setup. The retained interface is causal and stable enough to use, but still canonical opaque/private-code.

Conclusion:

- Robust positive for 1000-step lower-LR post-supervision retention: all three no-aux, no-anchor, frozen-upstream replications beat the drift control, and at least one improves on the prior C-low-lr result.
- Not a stronger protocol result: classification remains `causally_useful_opaque_private_code`, learned-best action-loss fraction remains `0.0`, and simple mapping explains only a small part of the gap.
- Recommendation: no-go on upstream unfreezing. Continue with input-proj-only stabilization, early-selection/stop criteria, and protocol-decoding probes before relaxing the frozen-upstream constraint.

## Action-loss-aligned self-training under the retention window

- Date: 2026-05-03.
- Task: `aiAgentProjectTasks/2026-05-03-phase-2-seventh-task-Action-loss-aligned-self-training-under-retention-window.md`.
- Work history: `aiAgentWorkHistory/phase2/2026-05-03-action-loss-aligned-self-training-retention-window.md`.
- Code changes: added `--checkpoint-every`, added `calculator_estimator=action_loss_weighted_interface`, and added `scripts/run_action_loss_candidate_diagnostic.py`.
- Primary constraints stayed strict: `aux_operand_loss_weight=0.0`, `input_proj_anchor_weight=0.0`, `freeze_upstream_encoder=true`, and only `calculator_hook.input_proj` trainable for dense and self-training primary runs.

Dense checkpoint-selection result:

| Dense run | Final eval exact | Best built-in normal snapshot | Final action gap | Action-loss-selected snapshot | Selected action gap | Selected operand/calc |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| Seed1 (`1 -> 3`) | `0.4688` | step `550`, normal `0.5781` | `2.8325` | step `100` | `2.1383` | `0.5625 / 0.5938` |
| Seed2 (`2 -> 4`) | `0.4707` | step `450`, normal `0.5938` | `2.8509` | step `550` | `2.2018` | `0.5469 / 0.6094` |
| Seed3 (`3 -> 5`) | `0.4180` | step `1300`, normal `0.5547` | `3.3582` | step `1050` | `2.4347` | `0.5625 / 0.5938` |

Checkpoint selection interpretation:

- Dense snapshot selection beats final-step selection in all three runs by built-in normal exact and by canonical action-loss learned-minus-true gap.
- Built-in normal-exact selection and action-loss-gap selection are only partly aligned, so future retention work should save weights densely and select by the metric needed for the claim.
- The selected dense checkpoint diagnosed canonically still has injection-zero `0.0000`, oracle-at-eval `0.9063`, true-sum-best forced-result fraction `0.9219`, and classification `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated`.

Candidate-action diagnostic:

| Checkpoint | Candidate better fraction | Mean best improvement | Candidate best result acc |
| --- | ---: | ---: | ---: |
| Stage B handoff | `0.570` | `2.8486` | `0.914` |
| Best prior lower-LR replication | `0.508` | `2.2891` | `0.883` |
| Dense selected examples | `0.523` to `0.570` | `2.7905` to `3.5481` | `0.852` to `0.891` |

Action-loss self-training result:

| Run | Final eval exact | Action learned-true gap | Operand exact | Calc result acc | Learned best | Classification if diagnosed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Selftrain seed1 (`1 -> 3`) | `0.4941` | `3.0137` | `0.4531` | `0.5000` | `0.0000` | n/a |
| Selftrain seed2 (`2 -> 4`) | `0.5449` | `1.9814` | `0.6094` | `0.6406` | `0.0000` | `semantically_decodable_private_calculator_code` |
| Selftrain seed3 (`3 -> 5`) | `0.4102` | `3.2365` | `0.4375` | `0.4844` | `0.0000` | n/a |

Best self-training private-protocol result:

- All-pair answer exact `0.5375`, operand exact `0.5500`, calculator-result accuracy `0.5775`.
- Best affine mapping is still identity-like: A exact `0.9000`, B exact `0.6025`.
- Group behavior remains uneven: no-carry operand exact `0.6364`, small-operands `0.5900`, carry `0.5362`, large-operand `0.5367`.
- Canonical causal controls remained good: injection-zero `0.0000`, forced-random `0.0625`, oracle-at-eval `0.9063`, true-sum-best forced-result fraction `0.9219`.

Decision:

- Checkpoint selection is a clear positive and should be part of future Stage C retention workflows.
- Candidate actions contain real answer-NLL signal, so this is not primarily candidate-limited at the tested pool size.
- The first action-loss-weighted self-training objective is promising but not robust: one of three seeds clearly improved action gap and operand/result behavior, two did not, and learned-best fraction stayed `0.0`.
- Recommendation remains no-go for upstream unfreezing. Next work should improve the action-loss objective under frozen-upstream/input-proj-only constraints, preferably using lower-variance targets, replay/selection, or continuing from action-loss-selected snapshots.
