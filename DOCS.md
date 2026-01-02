# Documentation: XOR Representation Learning Experiment

## Project Structure

```
├── main.py              # Entry point: parse config, run experiment
├── config.py            # Dataclass for hyperparameters, YAML loading
├── model.py             # MLP definition
├── task.py              # XOR dataset generation, train/val split
├── optimizers.py        # Adam, CMA-ES, HessianFree wrappers
├── training.py          # Training loop with noise averaging
├── metrics.py           # Correlation tracking, gradient stats
├── visualization.py     # Heatmap generation for TensorBoard
├── tests/
│   ├── conftest.py      # Shared fixtures
│   ├── test_task.py     # XOR correctness, train/val no overlap
│   ├── test_cmaes.py    # CMA-ES on toy problems
│   ├── test_correlations.py  # Correlation computation
│   └── test_noise_averaging.py
├── configs/
│   └── default.yaml     # Default configuration
└── runs/                # Experiment outputs (generated)
    └── {name}_{timestamp}/
        ├── config.yaml  # Copy of config used
        ├── tensorboard/ # TensorBoard logs
        └── checkpoint.pt # Final model weights
```

## Configuration (YAML)

```yaml
# Task
task:
  n: 10                  # Number of boolean input variables
  k: 5                   # XOR of first k variables
  train_fraction: 0.8    # Fraction for training (rest is validation)
  seed: 42               # Random seed for split

# Architecture
model:
  widths: [64, 64]       # Hidden layer widths (length = depth - 1)
  activation: relu       # relu, tanh, sigmoid

# Optimization
optimizer:
  type: adam             # adam, cmaes, hessianfree
  lr: 0.001              # Learning rate (Adam, HessianFree)
  # CMA-ES specific
  sigma0: 1.0            # Initial step size
  # HessianFree specific (TBD)

# Noise averaging
noise:
  enabled: true
  std: 0.1               # Gaussian noise standard deviation
  K: 16                  # Number of samples for gradient averaging

# Training
training:
  max_steps: 10000
  eval_every: 100        # Evaluate and log every N steps
  patience: 1000         # Early stopping patience (steps without improvement)

# Logging
logging:
  run_name: experiment   # Name prefix for run directory
  log_correlations_every: 500  # Log correlation heatmap every N steps
  gradient_history: 10   # Number of steps for gradient consistency
```

## Module Descriptions

### `config.py`
- `Config` dataclass with nested structure matching YAML
- `load_config(path: str) -> Config`: load and validate YAML
- CLI override support (optional)

### `model.py`
- `MLP(input_dim, widths, output_dim, activation) -> nn.Module`
- Simple sequential model with configurable depth/width/activation
- Method to get intermediate activations for correlation tracking

### `task.py`
- `xor_of_subset(x: Tensor, k: int) -> Tensor`: compute XOR of first k bits
- `make_dataset(n, k, train_fraction, seed) -> (train_x, train_y, val_x, val_y)`
- Generates all 2^n inputs, computes labels, splits without overlap

### `optimizers.py`
- `get_optimizer(model, config) -> OptimizerWrapper`
- Unified interface for all optimizers:
  ```python
  class OptimizerWrapper:
      def step(self, loss_fn: Callable) -> float:
          """Perform one optimization step, return loss value."""
  ```
- **Adam**: standard PyTorch, wraps `loss.backward()` + `optimizer.step()`
- **CMA-ES**: uses `pycma`, flatten/unflatten parameters
- **HessianFree**: uses PyTorchHessianFree, requires forward callback:
  ```python
  def forward():
      outputs = model(inputs)
      loss = loss_fn(outputs, targets)
      return loss, outputs  # Must return (loss, outputs) tuple
  loss_value = optimizer.step(forward=forward)
  ```

### `training.py`
- `train(model, train_data, val_data, optimizer, config, logger) -> dict`
- Main training loop
- Noise averaging: if `config.noise.enabled`:
  ```python
  grads = []
  for _ in range(K):
      noisy_params = params + noise
      loss = forward(noisy_params)
      grads.append(compute_grad(loss))
  avg_grad = mean(grads)
  ```
- Calls metrics/logging at appropriate intervals
- Early stopping based on validation accuracy

### `metrics.py`
- `compute_correlations(model, data, k) -> dict`
  - For each neuron in each layer, compute correlation with all 2^k XOR subsets
  - Return best subset and correlation value per neuron
- `gradient_stats(model, history) -> dict`
  - Gradient norms per layer
  - Sign consistency over last t steps

### `visualization.py`
- `correlation_heatmap(correlations, layer_sizes) -> PIL.Image`
  - Rows = neurons (grouped by layer, with separators)
  - Color = correlation strength
  - Annotation = best subset (as bit pattern or tuple)
- Returns image suitable for TensorBoard

## Libraries

| Purpose | Library | Install |
|---------|---------|---------|
| Neural networks | PyTorch | `pip install torch` |
| CMA-ES | pycma | `pip install cma` |
| HessianFree | PyTorchHessianFree | `pip install git+https://github.com/ltatzel/PyTorchHessianFree.git@main` |
| Logging | TensorBoard | `pip install tensorboard` |
| Config | PyYAML | `pip install pyyaml` |
| Testing | pytest | `pip install pytest` |

## Setup

```bash
# Install dependencies
uv sync

# Or add new dependency
uv add <package>
```

## Running Experiments

```bash
# Single run
uv run python main.py --config configs/default.yaml

# Override parameters
uv run python main.py --config configs/default.yaml --optimizer.type cmaes --model.widths [128,128]

# Run tests
uv run pytest tests/
```

## TensorBoard Logs

```bash
tensorboard --logdir runs/
```

Logged values:
- `loss/train`, `loss/val`: scalar losses
- `accuracy/train`, `accuracy/val`: scalar accuracies
- `gradient/norm_layer_{i}`: gradient norms per layer
- `gradient/sign_consistency`: mean sign consistency
- `correlations`: heatmap image (every N steps)

## Extending

### Adding a new optimizer
1. Add wrapper class in `optimizers.py` implementing `step(loss_fn) -> float`
2. Add config options to YAML schema
3. Update `get_optimizer()` factory function

### Adding a new metric
1. Add computation function in `metrics.py`
2. Call it in training loop (`training.py`)
3. Log to TensorBoard with appropriate tag
