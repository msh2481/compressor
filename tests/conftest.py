import pytest
import torch


@pytest.fixture
def seed():
    torch.manual_seed(42)
    return 42
