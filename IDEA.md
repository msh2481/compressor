# Research Idea: What Enables Deep Networks to Learn Representations?

## Core Question

What mechanisms allow neural networks trained with gradient descent to learn compositional structure (like XOR/parity) that kernel methods and local smoothing approaches (KNN, random forests, kernel ridge regression) provably cannot?

## Task

- **Input**: n boolean variables
- **Output**: XOR of the first k variables
- **Why this task**:
  - Canonical "hard" problem with known information-theoretic structure
  - Requires learning sparse compositional structure
  - Parity is provably hard for kernel methods / random features (requires exponential features)
  - Cleanly separates representational capacity from optimization dynamics

## Hypotheses

### H1: Width enables lottery ticket initialization
- Wide layers → higher probability that some neuron's random weights correlate with useful features (e.g., XOR of some subset)
- GD then amplifies these correlations and suppresses noise
- Prediction: wider networks succeed more often; initial correlation with target predicts success

### H2: Noise/dropout smooths the loss landscape
- With Gaussian noise injection + gradient averaging over K samples, we approximate gradient of RBF-smoothed loss
- This provides information about a *neighborhood*, not just the local point
- Effectively trades combinatorial search over dropout masks for combinatorial search over weight space
- Prediction: higher K → more configs find solution; phase transition at some critical K

### H3: Local vs global optimization
- First-order methods (Adam) are trapped by local structure
- Second-order methods (HessianFree) use curvature but still local
- Zero-order methods (CMA-ES) search globally
- Prediction: CMA-ES may succeed where Adam fails, indicating local minima are the bottleneck

## Experimental Design

### Independent Variables

| Variable | Values |
|----------|--------|
| Width | [16, 32, 64, 128, ...] |
| Depth | [2, 3, 4, ...] |
| Activation | [ReLU, Tanh, ...] |
| Optimizer | Adam, HessianFree, CMA-ES |
| Noise samples K | [1, 4, 16, 64, ...] |
| Task (n, k) | (6,3), (8,4), (10,5), ... |

### Dependent Variables / Metrics

1. **Accuracy**: train and validation (80/20 split of 2^n inputs, no overlap)
2. **Correlation tracking**: for each neuron, which XOR subset it correlates most with, and how strongly (visualized as image, grouped by layer)
3. **Gradient norms**: per layer
4. **Gradient consistency**: sign consistency across last t steps (per parameter)
5. **Loss landscape**: (optional) Hessian spectrum, loss barriers

### Key Comparisons

1. **Width sweep** at fixed depth/optimizer: test lottery ticket hypothesis
2. **K sweep** at fixed architecture: test noise smoothing hypothesis
3. **Optimizer comparison** at fixed architecture: test local vs global hypothesis
4. **Correlation tracking over time**: understand learning dynamics, order of feature discovery

## Expected Outcomes

- Identify configurations where each optimizer succeeds/fails
- Understand whether width helps via lottery ticket mechanism (measurable via initial correlations)
- Quantify benefit of gradient averaging (critical K for success)
- Characterize what makes XOR learnable: architecture, optimization, or their interaction

## Baselines

1. **Linear readout on random features**: random init network, train only last layer
2. **Kernel methods**: for comparison, how well does RBF kernel / random forest do?

## Open Questions

- Does depth matter independently of width for this task?
- Is there a phase transition in K (noise samples) for learnability?
- Can we predict success from initial neuron correlations?
