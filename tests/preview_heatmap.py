#!/usr/bin/env python3
"""
Quick preview of correlation heatmap without running full experiment.

Usage:
    uv run python tests/preview_heatmap.py
    uv run python tests/preview_heatmap.py --output heatmap.png
    uv run python tests/preview_heatmap.py --widths 16,16,8 --n 6
"""
import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import MLP
from metrics import compute_correlations
from visualization import correlation_heatmap


def main():
    parser = argparse.ArgumentParser(description="Preview correlation heatmap")
    parser.add_argument("--output", "-o", type=str, default="heatmap_preview.png",
                        help="Output image path")
    parser.add_argument("--widths", type=str, default="8,8",
                        help="Comma-separated layer widths")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of input bits")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "tanh", "sigmoid"])
    args = parser.parse_args()

    widths = [int(w) for w in args.widths.split(",")]

    print(f"Creating MLP: n={args.n}, widths={widths}, activation={args.activation}")
    model = MLP(
        input_dim=args.n,
        widths=widths,
        output_dim=1,
        activation=args.activation,
    )

    print(f"Computing correlations for {sum(widths)} neurons...")
    correlations = compute_correlations(model, n=args.n)

    print(f"Generating heatmap...")
    img = correlation_heatmap(correlations, layer_sizes=widths)

    output_path = Path(args.output)
    img.save(output_path)
    print(f"Saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
