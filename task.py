import torch
from torch import Tensor


def all_boolean_inputs(n: int) -> Tensor:
    """Generate all 2^n boolean input combinations."""
    return ((torch.arange(2**n).unsqueeze(1) >> torch.arange(n)) & 1).float()


def xor_of_subset(x: Tensor, k: int) -> Tensor:
    """Compute XOR of first k bits for each input row."""
    # x: (batch, n) with values in {0, 1}
    # Returns: (batch,) with values in {0, 1}
    return (x[:, :k].sum(dim=1) % 2).float()


def make_dataset(
    n: int, k: int, train_fraction: float = 0.8, seed: int = 42
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Generate XOR dataset with train/val split.

    Returns: (train_x, train_y, val_x, val_y)
    All tensors are float32.
    """
    all_x = all_boolean_inputs(n)
    all_y = xor_of_subset(all_x, k)

    # Shuffle and split
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(all_x), generator=generator)

    split_idx = int(len(all_x) * train_fraction)
    train_idx = perm[:split_idx]
    val_idx = perm[split_idx:]

    return all_x[train_idx], all_y[train_idx], all_x[val_idx], all_y[val_idx]


def get_all_xor_subsets(n: int) -> dict[tuple[int, ...], Tensor]:
    """
    Precompute XOR values for all non-empty subsets of n bits.

    Returns dict mapping subset tuple -> labels for all 2^n inputs.
    E.g., (0, 2) -> XOR of bits 0 and 2.
    """
    all_x = all_boolean_inputs(n)
    subsets = {}

    for mask in range(1, 2**n):
        bits = tuple(i for i in range(n) if mask & (1 << i))
        xor_val = (all_x[:, list(bits)].sum(dim=1) % 2).float()
        subsets[bits] = xor_val

    return subsets
