# Calculator-In-A-Model Experiment Fact Sheet

Last updated: 2026-04-30.

This document aggregates facts from `aiAgentWorkHistory/` and the completed task queue under `aiAgentProjectTasks/completed/`. It is deliberately scoped to facts from the specific experiments already run in this repo. It does not claim broad conclusions about calculators in neural networks beyond these implementations, tasks, architectures, and training settings.

## Project Frame

- The current testbed is synthetic addition as next-token prediction over custom digit/operator tokens, not natural language and not tool-call generation.
- Model A is the raw tiny causal transformer baseline.
- Model B wires the calculator hook but leaves the calculator off/zeroed.
- Model C enables the internal calculator path.
- The calculator interface is a residual-stream module: it projects hidden states to operand logits, discretizes operands, computes a hard sum internally, projects the result class back to the residual dimension, and injects at calculator positions.
- Most early experiments used additive residual injection at the `=` token: `h = h + calculator_injection`.
- Later experiments added operand-token read positions and a replacement injection mode, but the documented replacement mode still allowed autoregressive leakage through later answer-token residual context.

## Established Facts

### Data, Model, and Baselines

- The data generator emits strings like `a+b=sum<eos>` and later added `<pad>` plus fixed-width operands such as `93+09=102<eos>`.
- The batch pipeline shifts fixed-width sequences into next-token `x`/`y` tensors and masks loss only after `=`, including `<eos>`.
- A padding bug was fixed early: padded loss-mask positions must be `0`, not `PAD_ID`.
- The initial Model A implementation was a nanoGPT-style decoder-only transformer with default `n_layer=4`, `n_head=4`, `n_embd=128`, `dropout=0.0`, about `799K` params.
- The one-batch smoke test overfit 8 fixed 2-digit examples: loss went from about `2.65` to `0.0007` by step 50 and to `0.0000` by step 200, with final `8/8` accuracy.
- Baseline run `runs/2026-04-28_185900_model-a-baseline` gave:
  - 1-digit: `228/256 = 0.890625`, final loss `0.4046`.
  - 2-digit: `22/256 = 0.085938`, final loss `0.7580`.
  - 3-digit: `1/256 = 0.003906`, final loss `1.0929`.

### Latent Calculator Hook and First Model C Runs

- The latent calculator hook added `CalculatorHook` and `HardAddSTE` in `src/model.py`.
- In the first hook implementation, the hook read the `=` residual, projected to two latent operands, hard-discretized, summed internally, projected the result class, and added the injection only at `=` positions.
- Verification after the hook landed: `13` tests passed.
- Full Model B run `runs/2026-04-29_083546_model-b`:
  - 1-digit: `251/256 = 0.980`, final loss `0.0581`.
  - 2-digit: `24/256 = 0.094`, final loss `0.8049`.
  - 3-digit: `3/256 = 0.012`, final loss `1.0732`.
- Full Model C run `runs/2026-04-29_084448_model-c`:
  - 1-digit: `17/256 = 0.066`, final loss `1.6177`.
  - 2-digit: `0/256 = 0.000`, final loss `1.8488`.
  - 3-digit: `0/256 = 0.000`, final loss `1.9200`.
- In that first Model C setting, the active calculator path was harmful relative to Model B rather than helpful.

### Oracle and Diagnostic Facts

- `scripts/diagnose_calculator_protocol.py` added trace rows for predicted operands, confidences, entropies, result class, injection norms, `=` location, and oracle operand status.
- Oracle operands bypass the learned input projection while preserving the calculator result projection and downstream transformer path.
- Existing failed Model C checkpoint, learned operands:
  - exact match `0/16 = 0.000`.
  - operand exact match `0.000`.
  - calculator result accuracy `0.000`.
- Same failed checkpoint with oracle operands at eval:
  - exact match `0/16 = 0.000`.
  - operand exact match `1.000`.
  - calculator result accuracy `1.000`.
- Oracle-trained Model C `runs/2026-04-29_104845_model-c-oracle/model-c-1digit-seed1`:
  - 1-digit exact match `236/256 = 0.922`.
  - final loss `0.0648`.
  - oracle diagnostic exact match `59/64 = 0.922`.
  - oracle diagnostic operand exact match and calculator result accuracy both `1.000`.
- On the oracle-trained checkpoint evaluated with learned operands, exact match was `34/64 = 0.531`, operand exact match `0.000`, and calculator result accuracy `0.094`.
- These facts show that, in this implementation, the result-output/downstream path can be learned when correct operands are supplied during training. They do not show that the learned operand interface works.

### Tiny Operand Ranges and Auxiliary Operand Supervision

- Adding `--operand-max` and `--calculator-operand-vocab-size` made true tiny-vocabulary calculator interfaces possible.
- With hard STE and answer loss only:
  - `0..1`: eval `256/256 = 1.000`, diagnostic `64/64 = 1.000`, operand exact `0.203`, calc result `0.203`.
  - `0..2`: eval `256/256 = 1.000`, diagnostic `64/64 = 1.000`, operand exact `0.375`, calc result `0.438`.
  - `0..4`: eval `256/256 = 1.000`, diagnostic `64/64 = 1.000`, operand exact `0.000`, calc result `0.000`.
- A representative `0..4` trace showed `4+2=` answered correctly while sending `a_pred=3`, `b_pred=2`, result `5`.
- Residual probes showed the information was often present even when the hook did not use it:
  - on `0..1` and `0..2`, layer-2 `=` probes recovered both operands at `1.000 / 1.000`;
  - on `0..4`, layer-2 `=` probes reached `0.846 / 0.923` while hook operand exact was `0.000`.
- Auxiliary operand loss at `0..4`, weight `0.1`, produced eval `256/256 = 1.000`, operand exact `0.938`, and calc result accuracy `0.938`.
- Full `0..9` with aux `0.1` gave eval `221/256 = 0.863`, diagnostic `50/64 = 0.781`, operand exact `0.141`, calc result `0.141`.
- Full `0..9` with aux `1.0` gave eval `17/256 = 0.066`, diagnostic `2/64 = 0.031`, operand exact `0.016`, calc result `0.063`; one checked metrics JSON showed final loss `887.6017`.
- In these runs, auxiliary supervision demonstrated that the input projection can learn the intended small-range protocol in at least the `0..4` setting, but the tested full-range aux settings did not yield a reliable `0..9` protocol.

### REINFORCE / Sampled Calculator Actions

- `calculator_estimator=reinforce` samples A/B operands from categorical distributions and uses a score-function term with per-example masked answer loss.
- The policy loss used `mean((per_example_answer_loss.detach() - baseline) * sampled_logp_at_equals)`.
- A one-step gradient check found finite nonzero gradient into the calculator input projection, `input_proj_grad_norm ~= 0.00859`.
- Single-sample REINFORCE results:
  - `0..1`, seed 1: eval `256/256 = 1.000`, operand exact `0.242`, calc result `0.242`.
  - `0..1`, seed 2: eval `256/256 = 1.000`, operand exact `0.000`, calc result `0.000`.
  - `0..2`: eval `256/256 = 1.000`, operand exact `0.141`, calc result `0.258`.
  - `0..4`: eval `256/256 = 1.000`, operand exact `0.055`, calc result `0.148`.
  - `0..4` with aux `0.1` decayed over 500 steps: eval `256/256 = 1.000`, operand exact `0.047`, calc result `0.086`.
- In these exact settings, single-sample REINFORCE did not discover or maintain the intended true-operand protocol, even though the gradient plumbing reached the input projection.

### Capacity Reduction and Private-Code Discovery

- Architecture controls were added: `--n-layer`, `--n-head`, `--n-embd`, `--mlp-expansion`, and `--calculator-hook-after-layer`.
- A Model A/B/C initialization confound was fixed: hook modules previously consumed RNG before final LM-head initialization, so Model B/off did not share core initialization with Model A.
- Tiny 1-digit Model A sweep, 1000 steps, batch 64, eval 512, MLP expansion 1:
  - `1L/2H/32d`: `1.000`.
  - `1L/1H/16d`: `0.967`.
  - `1L/1H/8d`: `1.000`.
  - `1L/1H/4d`, seed 1: `0.691`.
  - `1L/1H/4d`, seed 2: `0.463`.
- Matching `1L/1H/4d` Model B/off:
  - seed 1: `0.588`.
  - seed 2: `0.490`.
- `1L/1H/4d` Model C:
  - STE seed 1: eval `0.418`, diagnostic `0.414`, operand exact `0.008`, calc result `0.094`.
  - STE seed 2: eval `0.916`, diagnostic `0.867`, operand exact `0.023`, calc result `0.117`.
  - oracle operands seed 1: eval `0.975`, diagnostic `0.977`, operand exact/result `1.000`.
  - STE + aux `0.1`: eval `0.939`, diagnostic `0.898`, operand exact/result `0.008`.
- Layer-1 `=` residual probes in the 4-dim regime were weak:
  - Model A seed 1: A/B probe `0.173 / 0.154`.
  - Model B seed 1: `0.154 / 0.192`.
  - Model C seed 2: `0.212 / 0.308`.
- The `1L/1H/4d` setting made A/B weak, but it may also be too narrow for a clean operand interface.
- Strong weird-protocol checkpoint `runs/2026-04-29_125836_910885_model-c-op0-9-1/model-c-1digit-seed2/final_weights.pt`:
  - normal learned result exact `0.904`.
  - operand exact `0.014`.
  - calc result accuracy `0.146`.
  - injection scale `0`: exact `0.645`.
  - forced zero result: exact `0.605`.
  - plus-one result: exact `0.578`.
  - random result: exact `0.650`.
  - oracle true operands: exact `0.637`, operand/result `1.000`.
- That checkpoint emitted only result classes `{4: 32, 6: 109, 11: 371}` on 512 samples.
- Its learned result class had about `1.06` bits of mutual information with true sum, and the best single mapping from result class to true sum reached only `0.256`.
- The precise fact from this checkpoint is not “the calculator was bypassed”; it is that this specific tiny Model C used a causally useful non-human result code that did not match the intended true-operand/true-sum protocol.

### Forced Result Sweeps and Codebooks

- Forced result-class diagnostics added `--forced-calculator-result-class` and `--forced-result-sweep`.
- Sweep outputs include `forced_result_sweep.csv`, `forced_result_summary.json`, and `result_codebook.csv`.
- Strong 1-digit checkpoint:
  - normal exact `0.904`.
  - true operand exact `0.014`.
  - calculator result accuracy `0.146`.
  - learned class was best forced class on `0.244` of prompts.
  - true-sum class was best forced class on `0.049` of prompts.
  - mean learned-minus-true-sum target log probability was `+0.655`.
  - learned class vs true sum MI `1.062` bits; vs first answer token `0.973`; vs carry `0.291`.
  - learned class `4` represented true sum `5`; class `6` represented `0,1,2,3,4,6`; class `11` represented `7..18`.
- Weaker 1-digit checkpoint:
  - normal exact `0.418`.
  - learned class best forced `0.424`.
  - true-sum class best forced `0.037`.
  - learned-minus-true-sum target log probability `+0.423`.
  - MI with true sum `1.718` bits.
- These sweeps strengthen the private-code fact for these checkpoints: forcing the human true-sum class was usually worse than the checkpoint's learned class.

### 2-Digit Capacity Regimes

- Candidate A used fixed-width 2-digit `0..99`, `2L/1H/16d/mlp1`, hook after layer 1, 1000 steps, 512 eval samples, run seed `2`.
- Candidate A results:
  - Model C oracle: `0.988`, final loss `0.0049`.
  - Model A: `0.711`, final loss `0.1380`.
  - Model B: `0.787`, final loss `0.1632`.
  - Model C learned STE: `0.023`, final loss `1.3661`.
- Candidate A learned STE diagnostics:
  - diagnostic exact `0.008`.
  - operand exact `0.000`.
  - calc result accuracy `0.008`.
  - mean operand confidences `1.000 / 0.996`.
  - injection-zero, oracle-at-eval, and forced-zero exact matches were all around `0.010`.
- Candidate B used `2L/1H/32d/mlp1`; oracle reached `1.000`, while Model A was `0.086` and Model B `0.047` in the documented run.
- The documented 2-digit result does not show learned protocol success; it identifies Candidate A as a useful regime where oracle works and A/B are imperfect, while plain learned STE collapsed.

### Non-Bottleneck `0..19` and Read Position

- Initial `operand_max=19`, `2L/1H/16d/mlp1`, hook after layer 1, 1000 steps, 512 eval samples:
  - Model A `0.949`.
  - Model B `0.627`.
  - Model C learned STE `0.557`.
  - Model C oracle `1.000`.
  - Model C + aux `0.01` `0.777`.
  - Model C oracle warmup `100` + aux `0.01` `0.049`.
- In that non-bottleneck `0..19` pass, aux `0.01` improved answer exact to `0.777` but true operand exact was `0.000`; counterfactuals were injection-zero `0.742`, forced-zero `0.766`, and oracle-at-eval `0.680`.
- Probe finding on that pass: operands were linearly accessible at operand-token positions but weak at the `=` read position.
- `calculator_read_position=operands` was then added:
  - A logits read from the final A digit residual.
  - B logits read from the final B digit residual.
  - calculator result injection remained at `=`.
- First operand-read `operand_max=19` rung:
  - Model A `0.883`.
  - Model B `0.977`.
  - Model C learned `0.902`.
  - Model C oracle `0.998`.
- Learned Model C under operand-read:
  - operand exact `0.000`.
  - calc result accuracy about `0.031` to `0.035`.
  - injection-zero `0.844`.
  - forced-zero `0.898`.
  - forced-random `0.898`.
  - oracle-at-eval `0.930`.
- Oracle Model C under operand-read:
  - operand exact `1.000`.
  - calc result `1.000`.
  - injection-zero `0.047`.
  - forced-zero `0.008`.
  - forced-random `0.031`.
  - oracle-at-eval `1.000`.
- Operand-read was implemented correctly enough for the oracle path to be strongly calculator-dependent, but learned answer-loss Model C still did not learn the true protocol in the first `0..19` rung.

### Protocol Supervision and Replacement Mode

- Track 2 added `calculator_injection_mode=add|replace`.
- `add` preserves previous residual addition.
- `replace` replaces active `=` residual positions with calculator injection and leaves non-`=` positions unchanged.
- Track 2 validation reported `39` tests passed.
- Track 2 setup: `digits=2`, `operand_max=19`, `calculator_operand_vocab_size=20`, `2L/1H/16d/mlp1`, hook after layer 1, `calculator_read_position=operands`, seed argument `0`, run seed `2`, 1000 steps, 512 eval samples.
- Non-bottleneck additive runs:
  - answer-only Model C exact `0.785`, operand `0.000`, calc result `0.031`.
  - aux `0.003` exact `0.127`.
  - aux `0.01` exact `0.961`, operand `0.000`, calc result `0.031`.
  - aux `0.03` exact `0.738`.
  - aux decay schedules collapsed to exact `0.018`, `0.000`, and `0.004`.
- Additive aux `0.01` had high answer accuracy but high counterfactual accuracies, so in this setting it was bypass-compatible rather than evidence of true calculator use.
- Replacement-mode runs:
  - Model B/off replacement control exact `0.889`.
  - oracle Model C replacement exact `1.000`, operand `1.000`, calc result `1.000`.
  - answer-only Model C replacement exact `0.133`.
  - aux replacement variants exact `0.098`, `0.059`, `0.059`, `0.094`, and `0.062`.
- Oracle replacement counterfactuals:
  - zero injection `0.039`.
  - forced zero `0.000`.
  - forced random `0.047`.
  - oracle eval `1.000`.
- The replacement implementation was not a valid calculator-required bottleneck in autoregressive generation, because Model B/off replacement still reached `0.889`; later answer-token positions could exploit ordinary residual context.

### Track 3 Diagnostics and Classifications

- Track 3 diagnostics now write trace rows, diagnostic summaries, result and operand codebooks, counterfactual exact-match tables, and protocol/bottleneck classifications.
- Track 3 added tensor-valued forced result classes, batched 2-digit forced sweeps, and read-site interventions for swapping or corrupting only A/B calculator read vectors.
- Six priority checkpoints were classified with 64 diagnostic samples:
  - Additive answer-only learned Model C: exact `0.750`, operand `0.000`, calc result `0.031`, zero injection `0.766`, forced random `0.750`, classification `calculator_ignored_or_bypassed`, bottleneck `non_bottleneck_leaky_by_design`.
  - Additive high-answer aux `0.01` Model C: exact `0.875`, operand `0.016`, calc result `0.047`, zero injection `0.859`, forced random `0.875`, classification `calculator_ignored_or_bypassed`.
  - Replacement Model B/off leakage control: exact `0.844`, classification `calculator_ignored_or_bypassed`, bottleneck `invalid_or_leaky_bottleneck`.
  - Replacement oracle Model C control: exact `1.000`, operand `1.000`, calc result `1.000`, zero injection `0.047`, forced random `0.062`, classification `valid_oracle_calculator_use`, bottleneck `invalid_or_leaky_bottleneck`.
  - Replacement answer-only learned Model C: exact `0.094`, operand `0.000`, calc result `0.047`, classification `calculator_ignored_or_bypassed`.
  - Replacement aux decay transient candidate: exact `0.078`, operand `0.000`, calc result `0.031`, zero injection `0.016`, forced random `0.047`, classification `causally_useful_opaque_private_code`.

### Track 4 Action-Loss Sensitivity

- Track 4 added `scripts/run_track4_action_loss_diagnostic.py`.
- The diagnostic reuses the Track 3 six-checkpoint manifest and classification labels.
- For each prompt, it measures target answer negative log likelihood under:
  - the normal learned calculator action;
  - the learned operand pair forced back through the calculator;
  - the true operand pair;
  - a shuffled true operand pair from another prompt;
  - multiple random operand pairs.
- Outputs include `action_loss_rows.csv`, `prompt_action_loss_summary.csv`, and `action_loss_summary.json` under each checkpoint's `track4_action_loss/` directory.
- Full run command: `PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_track4_action_loss_diagnostic.py --samples 64 --random-actions 16`.
- Results on the six priority checkpoints:
  - Additive answer-only learned Model C: true NLL `0.2204`, random NLL `0.2145`, random-minus-true gap `-0.0060`, operand exact `0.000`.
  - Additive high-answer aux `0.01` Model C: true NLL `0.0858`, random NLL `0.1114`, gap `+0.0256`, operand exact `0.0156`.
  - Replacement Model B/off leakage control: true/random/normal NLL all `0.1553`, action-loss std `0.0000`.
  - Replacement oracle Model C control: true NLL `0.0000`, random NLL `9.3949`, gap `+9.3949`, operand exact `1.000`.
  - Replacement answer-only learned Model C: true NLL `1.0149`, random NLL `1.0214`, gap `+0.0065`, operand exact `0.000`.
  - Replacement aux decay transient candidate: true NLL `1.6634`, random NLL `1.5016`, gap `-0.1618`, operand exact `0.000`.
- The oracle replacement result is the clean positive control: when true operands are supplied and the downstream path has learned to depend on the calculator, random operand actions are catastrophically worse than true actions.
- The learned and bypass checkpoints do not show a reliable intended true-action advantage. This weakens the case for rushing to estimator complexity before a valid bottleneck exists.
- Track 4 does not validate the current replacement mode as a bottleneck; it inherits Track 3's `invalid_or_leaky_bottleneck` label for replacement checkpoints.

### Implementation Review Facts

- A review found no serious implementation bug invalidating the main recent conclusions.
- One real issue was fixed: `calculator_hook_after_layer=0` previously meant the calculator was enabled in config but never called. A layer-0 embedding-stream hook path and regression test were added.
- Review notes verified:
  - Model A is a plain causal transformer.
  - Model B wires the hook but leaves it off.
  - Model C injects a hard calculator result additively by default.
  - Oracle operands bypass only the learned input projection and preserve the output projection/downstream path.
  - Counterfactual result overrides and injection-scale-zero are appropriate causal-use checks for the implemented hooks.

## Approach Ledger

| Approach | Tried so far | Result in these experiments | Hope / what remains |
| --- | --- | --- | --- |
| Model A raw transformer baseline | 1/2/3-digit defaults; later tiny and 2-digit capacity sweeps | Establishes ordinary-transformer controls. Default 2/3-digit baselines were weak; some later small-range or tiny-capacity regimes were unexpectedly strong or unstable. | Keep as control for every architecture/regime. |
| Model B hook-off control | Default hook-off, tiny-capacity hook-off, replacement/off | Mostly behaved as wiring control after initialization fix, but replacement/off reached `0.889`, exposing bottleneck leakage. | Continue requiring Model B controls; for future bottlenecks, Model B/off should fail if calculator is truly required. |
| Additive Model C with hard STE from `=` | First Model C default; tiny operand ranges; 2-digit candidate | Often failed true protocol; default Model C was worse than Model B; tiny ranges could answer perfectly while sending wrong operands. | Low hope by itself for true-operand protocol under answer loss, but remains the baseline for comparison. |
| Oracle operands / oracle training | 1-digit oracle train; 2-digit oracle; operand-read oracle; replacement oracle | Repeatedly works when correct operands are supplied, including oracle replacement. | Strong positive control. Use before interpreting learned failures. |
| Diagnostics and trace rows | Added protocol traces, summaries, probes, codebooks, counterfactuals | Prevented answer accuracy from being mistaken for protocol learning. | Keep expanding where needed; diagnostics are now central infrastructure. |
| Injection-scale and result counterfactuals | Injection-zero, forced zero, plus-one, random | Revealed both bypass-like runs and the 1-digit private-code checkpoint. | Continue for any promising learned checkpoint. |
| Tiny operand curriculum (`0..1`, `0..2`, `0..4`, `0..9`) | Hard STE and REINFORCE runs | Answer solved tiny ranges without intended operands; aux could teach `0..4`. | Still useful for credit-assignment debugging, not sufficient as success metric. |
| Auxiliary operand supervision | Constant aux `0.1` on tiny ranges; `0.003/0.01/0.03`; decay/floor schedules | `0..4` aux `0.1` worked for true protocol; `0..9` and 2-digit `0..19` schedules did not produce stable true protocol; decay schedules collapsed. | Try gentler/lower-LR schedules only if clearly separated from emergent success; persistent supervision may be needed. |
| Aux warmup / aux decay | REINFORCE aux decay; Track 2 decay to zero/floor | Did not retain the intended protocol after aux was removed or decayed in tried settings. | Could revisit with better bottleneck or lower-variance estimator, but current evidence is negative for these schedules. |
| Oracle warmup then learned operands | Non-bottleneck `oracle_warmup=100` + aux `0.01` | Failed badly, ending around `0.049` exact in the documented `0..19` run. | Still listed as a future idea, but immediate handoff was unstable. Needs gentler handoff/teacher forcing if retried. |
| REINFORCE / sampled calculator actions | Single-sample moving-baseline estimator, entropy, aux decay | Did not discover true protocol; gradient path was connected. | Next useful step is multi-sample per-prompt diagnostics and per-example/leave-one-out baselines before NVIL/MuProp/REBAR/RELAX. |
| Capacity reduction / dumber models | 1-digit `1L/1H/4d`, 2-digit `2L/1H/16d` and `32d` | 4d made A/B weak but may be too narrow; 2-digit 16d had oracle success and imperfect A/B but learned STE collapsed. | 2-digit 16d remains a useful candidate; more seeds and better protocol-learning pressure may be warranted. |
| Private-code / non-human protocol analysis | Counterfactuals and forced-result sweep on 1-digit checkpoints | At least one tiny checkpoint causally used a coarse private result code; not intended true-operand protocol. | Worth studying as a separate phenomenon: train for useful private protocols vs human-readable operands. |
| Forced result-class sweep / codebooks | 1-digit full sweep, later 2-digit batched machinery | Showed learned class was often better than true-sum class for private-code checkpoint. | Apply to any future causally useful learned checkpoint. |
| Probe-guided read position | Probes showed operands at A/B token positions; implemented `calculator_read_position=operands` | Oracle operand-read worked; learned answer-loss still did not learn true protocol in first `0..19` rung. | Do not scale solely from first negative rung; combine with better bottleneck/training signal if retried. |
| Non-bottleneck additive residual injection | Default and Track 2 additive runs | Easy to bypass; high answer accuracy often survived zero/forced-random calculator interventions. | Useful realistic setting, but not valid evidence of calculator reliance without counterfactual dependence. |
| Replacement injection mode | Replaced `=` residual with injection | Oracle replacement worked; learned replacement failed; Model B/off replacement still high, so bottleneck leaked. | Replace-at-`=` alone is insufficient. Need stricter answer-token residual blocking or explicit decoder phase. |
| Strict calculator-required bottleneck | Proposed but not yet implemented beyond leaky replacement | Not established. Current replacement was classified invalid/leaky. | High-priority next approach: block answer-token bypass or provide decoder only calculator result plus minimal formatting/context. |
| Softer bottleneck / compressed context | Proposed if strict bottleneck removes too much context | Not yet tried. | Try if strict bottleneck or oracle bottleneck fails due to missing formatting/context. |
| Explicit decoder phase | Proposed as a stronger bottleneck option | Not yet tried. | Candidate for separating prompt encoding from answer decoding so calculator result is the arithmetic bridge. |
| Gumbel-Softmax / soft expected-sum / hard-forward-soft-backward | Mentioned in task docs as estimator variants | Not yet run in the documented histories. | Still on the plate after STE/REINFORCE diagnostics; compare only with protocol metrics, not answer accuracy alone. |
| Multi-sample action-loss diagnostics | Track 4 forced true/learned/shuffled/random operand actions on six priority checkpoints | Oracle replacement has a huge true-vs-random loss signal; learned/bypass checkpoints mostly do not show a reliable true-action advantage. Model B/off has zero action sensitivity. | Use this as the pre-estimator gate. Next run it under a valid strict bottleneck before comparing estimator families. |
| Control-variate estimators: NVIL, MuProp, REBAR, RELAX | Mentioned as later variants | Not yet tried. | Only worth trying if multi-sample diagnostics show calculator choices affect loss but vanilla baseline is too noisy. |
| Larger-scale progression to `operand_max=49/99`, 3-digit, natural language | Planned guardrailed escalation | Not reached as a justified next step from Track 1/2 results. | Do not scale until a smaller setting shows credible calculator-use signature. |

## Current Working Interpretation

- The calculator output path is usable in multiple concrete oracle settings.
- The learned input protocol is the recurring failure point in the settings tried so far.
- Answer accuracy alone is not a valid success metric for this project; it repeatedly hid bypass behavior or private-code behavior.
- Some learned checkpoints are not pure bypass: at least one tiny checkpoint causally used a non-human calculator-result code.
- The first replacement bottleneck did not make the calculator strictly required because autoregressive answer positions still carried bypass information.
- Track 4 shows that the downstream objective can strongly distinguish true from random operand actions in the oracle replacement control, but the learned/bypass checkpoints generally do not present the intended true-action loss advantage.
- The most important next technical distinction is between:
  - learning a human-readable true-operand protocol,
  - learning any causally useful private calculator protocol,
  - and merely solving around the calculator.

## Near-Term Work Still On The Plate

- Build a stricter calculator-required bottleneck where Model B/off cannot solve by ordinary residual context.
- Run oracle first under any new bottleneck; learned failures are uninterpretable if oracle fails.
- Run the Track 4 action-loss diagnostic under that new bottleneck before estimator sweeps; true actions should reduce loss relative to random and shuffled actions.
- If oracle bottleneck works, test learned STE and supervised/auxiliary variants under that stricter bottleneck.
- Compare estimator families only after the bottleneck passes Model B/off, oracle, counterfactual, and action-loss-sensitivity checks.
- Continue using forced-result sweeps, codebooks, read-site interventions, injection-zero, forced-zero, forced-random, and oracle-at-eval checks for every promising checkpoint.
- Keep `operand_max=49/99`, larger models, and natural-language arithmetic behind a credible calculator-use signature in the smaller regimes.
