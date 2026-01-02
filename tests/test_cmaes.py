import torch
import torch.nn as nn
from optimizers import CMAESWrapper, AdamWrapper, create_optimizer
from config import OptimizerConfig
from model import MLP


def test_cmaes_quadratic():
    """CMA-ES should minimize a simple quadratic."""
    # Simple 2D quadratic: f(x) = x[0]^2 + x[1]^2
    model = nn.Linear(1, 2, bias=False)
    model.weight.data = torch.tensor([[3.0], [4.0]])  # Start far from optimum

    optimizer = CMAESWrapper(model, sigma0=1.0)

    def loss_fn():
        # Just minimize sum of squared weights
        loss = (model.weight**2).sum()
        return loss, model.weight

    # Run optimization
    for _ in range(50):
        loss = optimizer.step(loss_fn)

    # Should be close to zero
    assert loss < 0.1, f"CMA-ES failed to minimize quadratic: loss={loss}"


def test_cmaes_tiny_xor():
    """CMA-ES should solve tiny XOR problem."""
    # XOR: 2 inputs, 1 output
    model = MLP(input_dim=2, widths=[4], output_dim=1, activation="tanh")

    optimizer = CMAESWrapper(model, sigma0=0.5)

    # XOR dataset
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).float()
    y = torch.tensor([0, 1, 1, 0]).float()

    def loss_fn():
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        return loss, pred

    # Run optimization
    for _ in range(100):
        loss = optimizer.step(loss_fn)
        if loss < 0.01:
            break

    # Check accuracy
    with torch.no_grad():
        pred = model(x)
        acc = ((pred > 0.5).float() == y).float().mean()

    assert acc >= 0.75, f"CMA-ES failed on tiny XOR: acc={acc}"


def test_adam_wrapper():
    """Adam wrapper should work correctly."""
    model = nn.Linear(2, 1, bias=False)
    model.weight.data = torch.tensor([[3.0, 4.0]])

    optimizer = AdamWrapper(model, lr=0.1)

    def loss_fn():
        loss = (model.weight**2).sum()
        return loss, model.weight

    initial_loss = loss_fn()[0].item()
    for _ in range(50):
        optimizer.step(loss_fn)
    final_loss = loss_fn()[0].item()

    assert final_loss < initial_loss, "Adam should reduce loss"


def test_create_optimizer_adam():
    model = nn.Linear(2, 1)
    config = OptimizerConfig(type="adam", lr=0.01)
    opt = create_optimizer(model, config)
    assert isinstance(opt, AdamWrapper)
    assert opt.get_lr() == 0.01


def test_create_optimizer_cmaes():
    model = nn.Linear(2, 1)
    config = OptimizerConfig(type="cmaes", sigma0=0.5)
    opt = create_optimizer(model, config)
    assert isinstance(opt, CMAESWrapper)
