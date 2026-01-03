#!/usr/bin/env python3
"""
XOR Representation Learning Experiment

Usage:
    uv run python main.py --config configs/default.yaml
    uv run python main.py --config configs/default.yaml --optimizer.type cmaes
"""
import argparse
import sys
from pathlib import Path

from config import load_config, Config
from training import train


def parse_args():
    parser = argparse.ArgumentParser(description="XOR representation learning experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")

    # Allow overriding any config value via CLI
    # Format: --section.key value
    parser.add_argument("--task.n", type=int, help="Number of input bits")
    parser.add_argument("--task.k", type=int, help="XOR of first k bits")
    parser.add_argument("--task.train_fraction", type=float)
    parser.add_argument("--task.seed", type=int)

    parser.add_argument("--model.widths", type=str, help="Hidden layer widths, e.g. '[64,64]'")
    parser.add_argument("--model.activation", type=str, choices=["relu", "tanh", "sigmoid"])

    parser.add_argument("--optimizer.type", type=str, choices=["adam", "cmaes"])
    parser.add_argument("--optimizer.lr", type=float)
    parser.add_argument("--optimizer.sigma0", type=float)

    parser.add_argument("--noise.enabled", type=lambda x: x.lower() == "true")
    parser.add_argument("--noise.std", type=float)
    parser.add_argument("--noise.K", type=int)

    parser.add_argument("--training.max_steps", type=int)
    parser.add_argument("--training.eval_every", type=int)
    parser.add_argument("--training.patience", type=int)

    parser.add_argument("--logging.run_name", type=str)
    parser.add_argument("--logging.log_correlations_every", type=int)

    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply CLI overrides to config."""
    # Task
    if args.__dict__.get("task.n") is not None:
        config.task.n = args.__dict__["task.n"]
    if args.__dict__.get("task.k") is not None:
        config.task.k = args.__dict__["task.k"]
    if args.__dict__.get("task.train_fraction") is not None:
        config.task.train_fraction = args.__dict__["task.train_fraction"]
    if args.__dict__.get("task.seed") is not None:
        config.task.seed = args.__dict__["task.seed"]

    # Model
    if args.__dict__.get("model.widths") is not None:
        import ast
        config.model.widths = ast.literal_eval(args.__dict__["model.widths"])
    if args.__dict__.get("model.activation") is not None:
        config.model.activation = args.__dict__["model.activation"]

    # Optimizer
    if args.__dict__.get("optimizer.type") is not None:
        config.optimizer.type = args.__dict__["optimizer.type"]
    if args.__dict__.get("optimizer.lr") is not None:
        config.optimizer.lr = args.__dict__["optimizer.lr"]
    if args.__dict__.get("optimizer.sigma0") is not None:
        config.optimizer.sigma0 = args.__dict__["optimizer.sigma0"]

    # Noise
    if args.__dict__.get("noise.enabled") is not None:
        config.noise.enabled = args.__dict__["noise.enabled"]
    if args.__dict__.get("noise.std") is not None:
        config.noise.std = args.__dict__["noise.std"]
    if args.__dict__.get("noise.K") is not None:
        config.noise.K = args.__dict__["noise.K"]

    # Training
    if args.__dict__.get("training.max_steps") is not None:
        config.training.max_steps = args.__dict__["training.max_steps"]
    if args.__dict__.get("training.eval_every") is not None:
        config.training.eval_every = args.__dict__["training.eval_every"]
    if args.__dict__.get("training.patience") is not None:
        config.training.patience = args.__dict__["training.patience"]

    # Logging
    if args.__dict__.get("logging.run_name") is not None:
        config.logging.run_name = args.__dict__["logging.run_name"]
    if args.__dict__.get("logging.log_correlations_every") is not None:
        config.logging.log_correlations_every = args.__dict__["logging.log_correlations_every"]

    return config


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config = apply_overrides(config, args)

    # Print config summary
    print("=" * 60)
    print("XOR Representation Learning Experiment")
    print("=" * 60)
    print(f"Task: n={config.task.n}, k={config.task.k}")
    print(f"Model: widths={config.model.widths}, activation={config.model.activation}")
    print(f"Optimizer: {config.optimizer.type}")
    print(f"Noise: enabled={config.noise.enabled}, K={config.noise.K}, std={config.noise.std}")
    print("=" * 60)

    # Run training
    result = train(config)

    # Print results
    print("=" * 60)
    print("Results:")
    print(f"  Run dir: {result['run_dir']}")
    print(f"  Best val accuracy: {result['best_val_acc']:.4f}")
    print(f"  Final step: {result['final_step']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
