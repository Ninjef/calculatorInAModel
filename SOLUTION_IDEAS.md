Straight Through Estimator was the easy option. We tried that and it didn't seem to be working too well. Other options we need to explore:

# Alternatives to the Straight-Through Estimator

Great follow-up — and the right question to ask, because STE is just the *cheapest* solution to a much more general problem. Let me first sharpen what the problem actually is, then give you the menu.

## First, clarify which neurons are hard to train

In your calculator setup there are two populations:

- **Downstream neurons** (receiving the calculator's output): these are *easy*. The calculator's output is just an input feature to them — backprop works normally as long as you treat that input as a constant (stop-gradient).
- **Upstream neurons** (the ones whose activations get fed *into* the calculator): these are the hard ones. The calculator blocks the gradient, so they have no learning signal telling them *how to encode their representation into something the calculator can use*.

Every method below is really about getting a learning signal to those upstream encoder neurons. STE is one answer. Here are the others.

---

## The full menu of alternatives

### 1. **REINFORCE / Score-function estimators (the unbiased gradient family)**

Treat the calculator's input as a *stochastic action* sampled from a distribution parameterized by the upstream neurons. Then the policy-gradient identity gives you:

$$\nabla_\theta \mathbb{E}[L] = \mathbb{E}[L \cdot \nabla_\theta \log p_\theta(a)]$$

You don't need the calculator to be differentiable — you just need to evaluate the loss after running it. This is **unbiased**, unlike STE. The catch is **enormous variance**, which is why people developed control-variate variants: NVIL, MuProp, REBAR, RELAX. If your calculator outputs are discrete (digits, tokens), this is the most theoretically clean option.

### 2. **Gumbel-Softmax / Concrete relaxation**

If the upstream neurons need to produce *discrete* inputs to the calculator (e.g., digit tokens), you can replace the discrete sampling with a continuous, temperature-controlled relaxation. You then anneal temperature → 0 during training so the forward pass eventually matches the discrete one. This is biased but lower-variance than REINFORCE. A clean fit for "encode a number into the calculator" tasks.

### 3. **Differentiable surrogate / "shadow" emulator**

Train a small neural network to *mimic* the calculator end-to-end, then use the surrogate during backward passes while using the real calculator on the forward pass. This is essentially the same idea as differentiable physics emulators in scientific ML. The surrogate provides smooth gradients; the real calculator provides exact outputs. The risk is gradient–output mismatch — your gradients point the encoder toward "what fools the surrogate," not "what the real calculator wants."

### 4. **Zeroth-order / finite-difference gradient estimation**

Query the calculator at perturbed inputs and estimate the gradient numerically:

$$\hat{g} \approx \frac{1}{\sigma} \mathbb{E}_\epsilon[L(x+\sigma\epsilon)\,\epsilon]$$

This is what FOGZO and Evolution Strategies (ES) do. Unbiased in expectation, dimension-scaling cost. Works for *any* black-box module, including non-differentiable ones. Salimans et al.'s ES paper showed this can train neural networks competitively if you parallelize aggressively.

### 5. **Reinforcement learning framing**

Take this seriously: a calculator wired into the network is structurally identical to a tool-using policy. Treat the upstream layer as a *policy* whose action is "the calculator query," and the downstream task loss as the reward. PPO, actor-critic, or even DQN-style methods all apply. This is the principled version of #1 with engineering polish.

### 6. **Synthetic gradients / Decoupled Neural Interfaces (DNI)**

Train a small auxiliary network to *predict* what the gradient at the calculator boundary should be, given the upstream activations. The encoder gets fed this predicted gradient instead of a real one. Jaderberg et al. showed this allows asynchronous training and works through non-differentiable modules. Conceptually: you're learning the gradient itself.

### 7. **Target propagation (and Difference Target Propagation)**

Instead of propagating *gradients* backward, propagate *target activations*. The layer above the calculator computes "the activation I would have wanted from the calculator's output," and the upstream encoder is trained to map onto an inverse of that target. No gradients need to flow through the calculator at all. Bengio and collaborators developed this as a biologically-plausible alternative to backprop, but it's natural fit for non-differentiable modules.

### 8. **Forward-Forward (Hinton, 2022)**

Eliminate backprop entirely. Each layer is trained with a *local* "goodness" objective — high activity for positive (real) data, low for negative (corrupted) data. Since there's no global backward pass, the calculator's non-differentiability is irrelevant. The downside: Forward-Forward is still under-developed for large models, and you'd be doing two research projects at once.

### 9. **Direct Feedback Alignment (DFA)**

Replace the transposed-weight feedback path of backprop with **fixed random matrices**. Surprisingly, networks still learn. For your case, you could route a random feedback projection from the loss directly to the upstream neurons, bypassing the calculator entirely. Cheap, parallelizable, and indifferent to what's in between.

### 10. **Equilibrium Propagation / Predictive Coding**

Energy-based learning rules where the network settles into an equilibrium and weights are updated based on activity differences between two phases. These don't rely on differentiable forward operators in the standard sense. They're more exotic and currently confined to small-scale experiments, but they're a real alternative.

### 11. **Two-phase / curriculum training**

Pragmatic engineering trick: train initially with a *fully differentiable* approximation of the calculator (e.g., a learned MLP). Once the encoder has learned a sensible interface, freeze the encoder and swap in the real calculator. Then fine-tune downstream layers only. This sidesteps the gradient problem entirely by not having it during the hard learning phase.

### 12. **Implicit differentiation**

If your "calculator" is actually defined by an optimization problem or fixed-point equation (Z3 solver, root-finder, LP), you can differentiate through the *solution* using the implicit function theorem — no need to differentiate the solver's internals. This is what OptNet, DEQ, and JAX's `implicit_diff` do. Doesn't help for true black-box programs, but it's the right tool if your "calculator" is actually a constrained optimizer.
