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

    def get_linear_layers(self) -> list[nn.Linear]:
        """Get all Linear layers (for freezing)."""
        return [layer for layer in self.layers if isinstance(layer, nn.Linear)]

    def freeze_layer(self, layer_idx: int) -> None:
        """Freeze a specific linear layer by index."""
        linear_layers = self.get_linear_layers()
        if 0 <= layer_idx < len(linear_layers):
            for param in linear_layers[layer_idx].parameters():
                param.requires_grad = False

    def unfreeze_layer(self, layer_idx: int) -> None:
        """Unfreeze a specific linear layer by index."""
        linear_layers = self.get_linear_layers()
        if 0 <= layer_idx < len(linear_layers):
            for param in linear_layers[layer_idx].parameters():
                param.requires_grad = True

    def toggle_layer(self, layer_idx: int) -> bool:
        """Toggle freeze state of a layer. Returns new frozen state."""
        linear_layers = self.get_linear_layers()
        if 0 <= layer_idx < len(linear_layers):
            layer = linear_layers[layer_idx]
            is_frozen = not layer.weight.requires_grad
            if is_frozen:
                self.unfreeze_layer(layer_idx)
                return False
            else:
                self.freeze_layer(layer_idx)
                return True
        return False

    def get_frozen_status(self) -> list[bool]:
        """Get frozen status for each linear layer."""
        return [not layer.weight.requires_grad for layer in self.get_linear_layers()]


def create_model(
    n: int,
    widths: list[int],
    activation: Literal["relu", "tanh", "sigmoid"] = "relu",
) -> MLP:
    """Create MLP for XOR task."""
    return MLP(input_dim=n, widths=widths, output_dim=1, activation=activation)
