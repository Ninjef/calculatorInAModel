# Overview

The REINFORCE calculator-action runs showed a recurring failure mode: the model can reach perfect answer accuracy on tiny operand ranges while sending useless or wrong operands to the calculator. That means the ordinary transformer path is still strong enough to solve around the calculator, so downstream answer loss provides weak or misleading pressure on the calculator interface.

The next strategy is to keep the arithmetic/interface simple, but reduce the transformer's bypass capacity so the calculator path becomes useful or necessary.

# Research question

Can a deliberately smaller/weaker Model C learn to rely on the calculator interface for `0..9` single-digit addition when an equally small Model A cannot solve the task well on its own?

This is a cleaner next diagnostic than jumping directly to double-digit math:

- It keeps the calculator operand interface at two 10-way choices.
- It avoids the sparse exploration problem of 100-way whole-number operand heads.
- It directly tests whether bypass capacity is hiding calculator-learning signal.

# Experiments

## 1. Tiny Model A capacity sweep

Train raw Model A on 1-digit addition, operands `0..9`, across a small architecture sweep.

Candidate configs:

- `n_layer=1`, `n_embd=32`, `n_head=2`;
- `n_layer=1`, `n_embd=64`, `n_head=2`;
- `n_layer=2`, `n_embd=32`, `n_head=2`;
- optionally reduce MLP expansion if the code supports it or if adding it is simple and well-scoped.

Goal:

Find the smallest architecture that does not trivially solve 1-digit addition by the ordinary transformer path, while still being trainable enough to parse the prompt.

Report:

- answer exact match;
- loss curves;
- representative predictions;
- parameter count.

## 2. Matching tiny Model B control

For any promising small architecture, train Model B with the calculator hook wired but off.

Decision rule:

- If Model B is much worse than Model A, debug hook/wiring/capacity effects before interpreting Model C.
- If Model B matches Model A, proceed.

## 3. Tiny Model C with calculator on

Train the same small architecture as Model C with the calculator enabled.

Try:

- hard STE;
- REINFORCE;
- auxiliary operand warmup plus decay;
- optionally constant auxiliary operand supervision as an upper-bound diagnostic.

Report:

- answer exact match;
- true operand exact match;
- calculator result accuracy;
- operand entropy/confidence;
- calculator trace rows;
- policy metrics for REINFORCE runs.

Decision rule:

- If tiny Model A/B fail but tiny Model C succeeds with high calculator-result accuracy, that is evidence the calculator can help when bypass capacity is constrained.
- If tiny Model C answers correctly but calculator traces remain bad, the model found another shortcut.
- If all variants fail, the architecture may be too weak to parse operands or use the injected result.
- If auxiliary warmup works but vanilla STE/REINFORCE does not, the bottleneck remains protocol discovery/credit assignment rather than calculator usefulness.

## 4. Placement diagnostic

For 1- or 2-layer models, calculator placement matters more than before.

Try hook positions that are legal for the depth:

- after layer 1 for `n_layer=1`;
- after layer 1 and after layer 2 for `n_layer=2`.

Interpretation:

- Too early may mean the upstream residual does not contain operand information.
- Too late may leave too little downstream capacity to use the injected result.

Use residual probes if the trace results are ambiguous.

# Important measurement warning

Do not treat answer accuracy alone as calculator success.

A run is successful only if at least one of these is true:

- true operand exact match is high;
- calculator result accuracy is high and trace rows show a coherent alternate protocol worth studying.

Tiny-model answer success with bad calculator traces is still a bypass/shortcut result.

# Implementation notes

The training script may need CLI flags for architecture parameters:

- `--n-layer`;
- `--n-head`;
- `--n-embd`;
- possibly `--calculator-hook-after-layer`.

Keep changes scoped and preserve existing defaults so prior runs remain reproducible.

If adding an MLP expansion factor, do it only if the architecture sweep shows the current tiny configs are still too capable or too weak in an unhelpful way.

# Acceptance criteria

Save for each experiment:

- config;
- final checkpoint;
- training curve;
- answer metrics;
- diagnostic summary;
- calculator trace rows;
- policy-gradient-specific metrics when applicable.

The next useful milestone is a short writeup answering:

- Does reducing transformer capacity make calculator use emerge?
- What is the smallest architecture where Model A/B struggle but Model C can succeed?
- Does Model C learn true operands, a coherent alternate protocol, or another shortcut?
- Which hook placement works best for tiny models?
- Does auxiliary warmup become maintainable when the calculator path is genuinely useful?
