import torch
from torch import Tensor
import torch.nn as nn
from collections import deque
from dataclasses import dataclass

from model import MLP
from task import get_all_xor_subsets, all_boolean_inputs


@dataclass
class NeuronCorrelation:
    """Correlation info for a single neuron."""
    layer: int
    neuron_idx: int
    best_subset: tuple[int, ...]
    correlation: float


def compute_correlations(
    model: MLP,
    n: int,
) -> list[NeuronCorrelation]:
    """
    Compute correlation of each neuron's activation with all XOR subsets.

    Returns list of NeuronCorrelation objects, one per neuron.
    """
    # Get all inputs and XOR subset labels
    all_x = all_boolean_inputs(n)
    xor_subsets = get_all_xor_subsets(n)

    # Get activations for all inputs
    with torch.no_grad():
        activations = model.get_activations(all_x)

    results = []

    for layer_idx, layer_act in enumerate(activations):
        # layer_act: (2^n, width)
        num_neurons = layer_act.shape[1]

        for neuron_idx in range(num_neurons):
            neuron_act = layer_act[:, neuron_idx]

            # Find best correlating XOR subset
            best_subset = None
            best_corr = -1.0

            for subset, xor_labels in xor_subsets.items():
                # Pearson correlation
                corr = _pearson_correlation(neuron_act, xor_labels)
                # Take absolute value (anti-correlation is also useful)
                if abs(corr) > best_corr:
                    best_corr = abs(corr)
                    best_subset = subset

            results.append(NeuronCorrelation(
                layer=layer_idx,
                neuron_idx=neuron_idx,
                best_subset=best_subset,
                correlation=best_corr,
            ))

    return results


def _pearson_correlation(x: Tensor, y: Tensor) -> float:
    """Compute Pearson correlation coefficient."""
    x = x - x.mean()
    y = y - y.mean()

    num = (x * y).sum()
    denom = (x.norm() * y.norm()) + 1e-8

    return (num / denom).item()


class GradientTracker:
    """Track gradient statistics (weights only, no biases)."""

    def __init__(self, model: nn.Module):
        self.model = model
        # Store previous gradient for cosine similarity
        self._prev_grads: dict[str, Tensor] = {}

    def get_norms_per_layer(self) -> dict[str, float]:
        """Get gradient norms per layer (weights only)."""
        norms = {}
        for name, p in self.model.named_parameters():
            if "bias" not in name and p.grad is not None:
                norms[name] = p.grad.norm().item()
        return norms

    def get_cosine_similarity(self) -> dict[str, float]:
        """
        Get cosine similarity between current and previous gradient.

        Returns value in [-1, 1], where 1 = same direction, -1 = opposite.
        """
        similarities = {}
        for name, p in self.model.named_parameters():
            if "bias" not in name and p.grad is not None:
                curr = p.grad.flatten()
                if name in self._prev_grads:
                    prev = self._prev_grads[name]
                    cos_sim = torch.nn.functional.cosine_similarity(
                        curr.unsqueeze(0), prev.unsqueeze(0)
                    ).item()
                    similarities[name] = cos_sim
                # Update stored gradient
                self._prev_grads[name] = curr.clone()
        return similarities


def compute_accuracy(model: nn.Module, x: Tensor, y: Tensor) -> float:
    """Compute classification accuracy."""
    with torch.no_grad():
        pred = model(x)
        pred_binary = (pred > 0.5).float()
        acc = (pred_binary == y).float().mean()
    return acc.item()
