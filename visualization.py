import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image

from metrics import NeuronCorrelation


def correlation_heatmap(
    correlations: list[NeuronCorrelation],
    layer_sizes: list[int],
) -> Image.Image:
    """
    Create table visualization of neuron correlations.

    Rows = layers
    Columns = neuron index within layer
    Cells = subset text, colored by correlation strength

    Returns PIL Image suitable for TensorBoard.
    """
    # Group correlations by layer
    by_layer: dict[int, list[NeuronCorrelation]] = {}
    for corr in correlations:
        by_layer.setdefault(corr.layer, []).append(corr)

    num_layers = len(by_layer)
    if num_layers == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No correlations", ha="center", va="center")
        ax.axis("off")
        return _fig_to_image(fig)

    # Find max width for table columns
    max_width = max(layer_sizes) if layer_sizes else 1

    # Cell dimensions
    cell_width = 1.0
    cell_height = 0.6

    fig_width = max(6, max_width * cell_width + 1.5)
    fig_height = max(3, num_layers * cell_height + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Draw table
    for layer_idx in sorted(by_layer.keys()):
        layer_corrs = sorted(by_layer[layer_idx], key=lambda c: c.neuron_idx)
        y = num_layers - 1 - layer_idx  # Flip so layer 0 is at top

        for corr in layer_corrs:
            x = corr.neuron_idx
            color_val = corr.correlation

            # Draw cell background
            rect = plt.Rectangle(
                (x * cell_width, y * cell_height),
                cell_width,
                cell_height,
                facecolor=plt.cm.viridis(color_val),
                edgecolor="white",
                linewidth=1,
            )
            ax.add_patch(rect)

            # Add subset text
            subset_str = _format_subset(corr.best_subset)
            text_color = "white" if color_val > 0.5 else "black"
            ax.text(
                x * cell_width + cell_width / 2,
                y * cell_height + cell_height / 2,
                subset_str,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    # Add layer labels on the left
    for layer_idx in range(num_layers):
        y = num_layers - 1 - layer_idx
        ax.text(
            -0.15,
            y * cell_height + cell_height / 2,
            f"L{layer_idx}",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Add neuron index labels on top
    for i in range(max_width):
        ax.text(
            i * cell_width + cell_width / 2,
            num_layers * cell_height + 0.1,
            str(i),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlim(-0.3, max_width * cell_width)
    ax.set_ylim(-0.2, num_layers * cell_height + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Neuron-XOR Subset Correlations", pad=10)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.02)
    cbar.set_label("Correlation", fontsize=9)

    plt.tight_layout()
    return _fig_to_image(fig)


def _format_subset(subset: tuple[int, ...]) -> str:
    """Format subset as tuple string like (0,2,3)."""
    if subset is None:
        return "?"
    return "(" + ",".join(map(str, subset)) + ")"


def _fig_to_image(fig: Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def dashboard_figure(
    steps: list[int],
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    grad_norms: dict[str, list[float]],
    grad_cosine: dict[str, list[float]],
) -> Image.Image:
    """
    Create 2x2 dashboard figure.

    Top-left: loss (train/val)
    Top-right: accuracy (train/val)
    Bottom-left: gradient norms per layer
    Bottom-right: gradient cosine similarity per layer
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top-left: Loss
    ax = axes[0, 0]
    ax.plot(steps, train_losses, label="train", alpha=0.8)
    ax.plot(steps, val_losses, label="val", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    if train_losses and max(train_losses) > 0:
        ax.set_yscale("log")
    if len(steps) > 1:
        ax.set_xscale("log")

    # Top-right: Accuracy
    ax = axes[0, 1]
    ax.plot(steps, train_accs, label="train", alpha=0.8)
    ax.plot(steps, val_accs, label="val", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.set_ylim(0, 1.05)
    if len(steps) > 1:
        ax.set_xscale("log")

    # Bottom-left: Gradient norms
    ax = axes[1, 0]
    for name, norms in grad_norms.items():
        # Shorten layer name for legend
        short_name = name.replace("layers.", "L").replace(".weight", "")
        ax.plot(steps[:len(norms)], norms, label=short_name, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Norm")
    ax.set_title("Gradient Norms")
    ax.legend(fontsize=8)
    if grad_norms and any(grad_norms.values()):
        ax.set_yscale("log")
    if len(steps) > 1:
        ax.set_xscale("log")

    # Bottom-right: Cosine similarity
    ax = axes[1, 1]
    # Offset steps by 1 since cosine needs previous gradient
    cosine_steps = steps[1:] if len(steps) > 1 else steps
    for name, cosines in grad_cosine.items():
        short_name = name.replace("layers.", "L").replace(".weight", "")
        ax.plot(cosine_steps[:len(cosines)], cosines, label=short_name, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Gradient Cosine Similarity")
    if grad_cosine:
        ax.legend(fontsize=8)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    if len(cosine_steps) > 1:
        ax.set_xscale("log")

    plt.tight_layout()
    return _fig_to_image(fig)
