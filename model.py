import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    """Simple MLP with configurable width, depth, and activation."""

    def __init__(
        self,
        input_dim: int,
        widths: list[int],
        output_dim: int = 1,
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.widths = widths
        self.output_dim = output_dim

        act_cls = ACTIVATIONS[activation]

        layers = []
        prev_dim = input_dim
        for width in widths:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(act_cls())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)

    def get_activations(self, x: Tensor) -> list[Tensor]:
        """
        Get activations after each hidden layer (post-activation).

        Returns list of tensors, one per hidden layer.
        """
        activations = []
        h = x
        for i, layer in enumerate(self.layers[:-1]):  # exclude final linear
            h = layer(h)
            # Collect activations after each activation function (odd indices)
            if i % 2 == 1:
                activations.append(h)
        return activations

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(
    n: int,
    widths: list[int],
    activation: Literal["relu", "tanh", "sigmoid"] = "relu",
) -> MLP:
    """Create MLP for XOR task."""
    return MLP(input_dim=n, widths=widths, output_dim=1, activation=activation)
