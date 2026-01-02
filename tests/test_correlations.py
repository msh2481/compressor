import torch
import torch.nn as nn
from model import MLP
from metrics import compute_correlations, _pearson_correlation, GradientTracker, compute_accuracy
from visualization import correlation_heatmap


def test_pearson_correlation_perfect():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y = torch.tensor([0.0, 1.0, 2.0, 3.0])
    corr = _pearson_correlation(x, y)
    assert abs(corr - 1.0) < 0.01


def test_pearson_correlation_anticorrelated():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y = torch.tensor([3.0, 2.0, 1.0, 0.0])
    corr = _pearson_correlation(x, y)
    assert abs(corr - (-1.0)) < 0.01


def test_pearson_correlation_uncorrelated():
    x = torch.tensor([0.0, 1.0, 0.0, 1.0])
    y = torch.tensor([0.0, 0.0, 1.0, 1.0])
    corr = _pearson_correlation(x, y)
    assert abs(corr) < 0.01


def test_compute_correlations_shape():
    model = MLP(input_dim=4, widths=[8, 8], output_dim=1, activation="relu")
    correlations = compute_correlations(model, n=4)

    # Should have 8 + 8 = 16 neurons
    assert len(correlations) == 16


def test_compute_correlations_layers():
    model = MLP(input_dim=4, widths=[4, 6], output_dim=1, activation="relu")
    correlations = compute_correlations(model, n=4)

    # Check layer assignments
    layer_0 = [c for c in correlations if c.layer == 0]
    layer_1 = [c for c in correlations if c.layer == 1]

    assert len(layer_0) == 4
    assert len(layer_1) == 6


def test_compute_correlations_values_in_range():
    model = MLP(input_dim=4, widths=[8], output_dim=1, activation="tanh")
    correlations = compute_correlations(model, n=4)

    for corr in correlations:
        assert 0 <= corr.correlation <= 1.0
        assert corr.best_subset is not None
        assert len(corr.best_subset) >= 1


def test_gradient_tracker():
    model = nn.Linear(4, 2)
    tracker = GradientTracker(model)

    # First iteration - no cosine sim yet (no previous gradient)
    x = torch.randn(8, 4)
    y = model(x).sum()
    y.backward()

    norms = tracker.get_norms_per_layer()
    assert len(norms) > 0

    cosines = tracker.get_cosine_similarity()
    assert len(cosines) == 0  # No previous gradient yet

    model.zero_grad()

    # Second iteration - should have cosine similarity
    x = torch.randn(8, 4)
    y = model(x).sum()
    y.backward()

    cosines = tracker.get_cosine_similarity()
    assert len(cosines) > 0
    for v in cosines.values():
        assert -1 <= v <= 1


def test_gradient_tracker_norms():
    model = nn.Linear(4, 2)
    tracker = GradientTracker(model)

    x = torch.randn(8, 4)
    y = model(x).sum()
    y.backward()

    norms = tracker.get_norms_per_layer()
    assert len(norms) > 0
    for v in norms.values():
        assert v >= 0


def test_compute_accuracy():
    model = MLP(input_dim=2, widths=[4], output_dim=1, activation="relu")

    # Simple case: all zeros
    x = torch.zeros(4, 2)
    y = torch.zeros(4)

    # Model output is unpredictable, but function should work
    acc = compute_accuracy(model, x, y)
    assert 0 <= acc <= 1


def test_correlation_heatmap_runs():
    model = MLP(input_dim=4, widths=[4, 4], output_dim=1, activation="relu")
    correlations = compute_correlations(model, n=4)

    img = correlation_heatmap(correlations, layer_sizes=[4, 4])

    # Should return a PIL Image
    assert img.mode in ["RGB", "RGBA"]
    assert img.size[0] > 0 and img.size[1] > 0


def test_correlation_heatmap_empty():
    img = correlation_heatmap([], layer_sizes=[])
    assert img.size[0] > 0
