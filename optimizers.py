from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import cma

from config import OptimizerConfig


class OptimizerWrapper(ABC):
    """Unified interface for all optimizers."""

    @abstractmethod
    def step(
        self,
        loss_fn: Callable[[], tuple[Tensor, Tensor]],
    ) -> float:
        """
        Perform one optimization step.

        Args:
            loss_fn: Callable that returns (loss, outputs).

        Returns:
            Loss value as float.
        """
        pass

    @abstractmethod
    def get_lr(self) -> float:
        """Get current learning rate (or equivalent)."""
        pass


class AdamWrapper(OptimizerWrapper):
    """Wrapper for PyTorch Adam optimizer."""

    def __init__(self, model: nn.Module, lr: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._lr = lr

    def step(self, loss_fn: Callable[[], tuple[Tensor, Tensor]]) -> float:
        self.optimizer.zero_grad()
        loss, _ = loss_fn()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_lr(self) -> float:
        return self._lr


class CMAESWrapper(OptimizerWrapper):
    """Wrapper for CMA-ES optimizer from pycma."""

    def __init__(self, model: nn.Module, sigma0: float = 1.0):
        self.model = model
        self.sigma0 = sigma0

        # Flatten initial parameters
        self._param_shapes = [p.shape for p in model.parameters()]
        self._param_sizes = [p.numel() for p in model.parameters()]
        x0 = self._flatten_params()

        # Initialize CMA-ES
        self.es = cma.CMAEvolutionStrategy(
            x0.tolist(),
            sigma0,
            {"verbose": -9},  # suppress output
        )

    def _flatten_params(self) -> np.ndarray:
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.model.parameters()])

    def _unflatten_params(self, flat: np.ndarray) -> None:
        offset = 0
        for p, shape, size in zip(self.model.parameters(), self._param_shapes, self._param_sizes):
            p.data = torch.from_numpy(flat[offset : offset + size].reshape(shape)).float()
            if p.is_cuda:
                p.data = p.data.cuda()
            offset += size

    def step(self, loss_fn: Callable[[], tuple[Tensor, Tensor]]) -> float:
        # Ask for candidate solutions
        solutions = self.es.ask()

        # Evaluate each candidate
        fitness = []
        for sol in solutions:
            self._unflatten_params(np.array(sol))
            with torch.no_grad():
                loss, _ = loss_fn()
            fitness.append(loss.item())

        # Tell CMA-ES the results
        self.es.tell(solutions, fitness)

        # Set parameters to current best
        self._unflatten_params(np.array(self.es.result.xbest))

        return min(fitness)

    def get_lr(self) -> float:
        return self.es.sigma


def create_optimizer(model: nn.Module, config: OptimizerConfig) -> OptimizerWrapper:
    """Factory function to create optimizer from config."""
    if config.type == "adam":
        return AdamWrapper(model, lr=config.lr)
    elif config.type == "cmaes":
        return CMAESWrapper(model, sigma0=config.sigma0)
    else:
        raise ValueError(f"Unknown optimizer type: {config.type}")
