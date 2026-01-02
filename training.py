from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from config import Config
from model import MLP, create_model
from task import make_dataset
from optimizers import OptimizerWrapper, create_optimizer
from metrics import compute_correlations, GradientTracker, compute_accuracy
from visualization import correlation_heatmap, dashboard_figure


def add_noise_to_params(model: nn.Module, std: float) -> dict[str, Tensor]:
    """Add Gaussian noise to parameters, return original values."""
    original = {}
    for name, p in model.named_parameters():
        original[name] = p.data.clone()
        p.data = p.data + torch.randn_like(p.data) * std
    return original


def restore_params(model: nn.Module, original: dict[str, Tensor]) -> None:
    """Restore parameters from saved values."""
    for name, p in model.named_parameters():
        p.data = original[name]


def noise_averaged_step(
    model: nn.Module,
    optimizer: OptimizerWrapper,
    loss_fn_factory,
    noise_std: float,
    K: int,
) -> float:
    """
    Perform optimization step with noise-averaged gradients.

    Args:
        model: The model to optimize
        optimizer: Optimizer wrapper
        loss_fn_factory: Callable that returns (loss, outputs) for current params
        noise_std: Standard deviation of Gaussian noise
        K: Number of noise samples

    Returns:
        Loss value
    """
    if K <= 1:
        # No noise averaging
        return optimizer.step(loss_fn_factory)

    # For Adam: manually average gradients
    if hasattr(optimizer, "optimizer"):  # AdamWrapper
        # Accumulate gradients over K noise samples
        accumulated_grads = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

        losses = []
        for _ in range(K):
            original = add_noise_to_params(model, noise_std)

            optimizer.optimizer.zero_grad()
            loss, _ = loss_fn_factory()
            loss.backward()
            losses.append(loss.item())

            # Accumulate
            for name, p in model.named_parameters():
                if p.grad is not None:
                    accumulated_grads[name] += p.grad

            restore_params(model, original)

        # Average gradients
        for name, p in model.named_parameters():
            p.grad = accumulated_grads[name] / K

        # Step with averaged gradients
        optimizer.optimizer.step()
        return np.mean(losses)

    else:
        # For CMA-ES / HessianFree: just call step (they handle it internally)
        return optimizer.step(loss_fn_factory)


def train(config: Config, run_dir: Path | None = None) -> dict:
    """
    Run training experiment.

    Args:
        config: Experiment configuration
        run_dir: Directory to save logs. If None, creates timestamped dir.

    Returns:
        Dict with final metrics and paths.
    """
    # Setup run directory
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"{config.logging.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    from config import save_config
    save_config(config, run_dir / "config.yaml")

    # Setup plots directory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Create dataset
    train_x, train_y, val_x, val_y = make_dataset(
        n=config.task.n,
        k=config.task.k,
        train_fraction=config.task.train_fraction,
        seed=config.task.seed,
    )

    # Create model
    model = create_model(
        n=config.task.n,
        widths=config.model.widths,
        activation=config.model.activation,
    )

    # Create optimizer
    optimizer = create_optimizer(model, config.optimizer)

    # Setup gradient tracker
    grad_tracker = GradientTracker(model)

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()

    def make_loss_fn():
        def loss_fn_inner():
            logits = model(train_x)
            loss = loss_fn(logits, train_y)
            return loss, logits
        return loss_fn_inner

    # Training state
    best_val_acc = 0.0
    steps_without_improvement = 0
    history = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "grad_norms": {},  # layer_name -> list of norms
        "grad_cosine": {},  # layer_name -> list of cosine similarities
    }

    # Training loop
    for step in range(config.training.max_steps):
        # Optimization step
        if config.noise.enabled and config.noise.K > 1:
            train_loss = noise_averaged_step(
                model, optimizer, make_loss_fn(),
                noise_std=config.noise.std,
                K=config.noise.K,
            )
        else:
            train_loss = optimizer.step(make_loss_fn())

        # Evaluation
        if step % config.training.eval_every == 0:
            with torch.no_grad():
                val_logits = model(val_x)
                val_loss = loss_fn(val_logits, val_y).item()

            train_acc = compute_accuracy(model, train_x, train_y)
            val_acc = compute_accuracy(model, val_x, val_y)

            # Record gradient stats (also updates prev gradient for cosine sim)
            cosines = grad_tracker.get_cosine_similarity()
            for name, cos in cosines.items():
                if name not in history["grad_cosine"]:
                    history["grad_cosine"][name] = []
                history["grad_cosine"][name].append(cos)

            norms = grad_tracker.get_norms_per_layer()
            for name, norm in norms.items():
                if name not in history["grad_norms"]:
                    history["grad_norms"][name] = []
                history["grad_norms"][name].append(norm)

            # Save history
            history["steps"].append(step)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                steps_without_improvement = 0
            else:
                steps_without_improvement += config.training.eval_every

            # Print progress
            print(f"Step {step}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

            # Early stopping
            if steps_without_improvement >= config.training.patience:
                print(f"Early stopping at step {step}")
                break

            # Perfect accuracy
            if val_acc >= 1.0 and train_acc >= 1.0:
                print(f"Perfect accuracy at step {step}")
                break

        # Save plots periodically
        if step % config.logging.log_correlations_every == 0:
            # Dashboard
            dashboard = dashboard_figure(
                steps=history["steps"],
                train_losses=history["train_loss"],
                val_losses=history["val_loss"],
                train_accs=history["train_acc"],
                val_accs=history["val_acc"],
                grad_norms=history["grad_norms"],
                grad_cosine=history["grad_cosine"],
            )
            dashboard.save(plots_dir / "dashboard.png")

            # Correlations
            correlations = compute_correlations(model, config.task.n)
            corr_img = correlation_heatmap(correlations, config.model.widths)
            corr_img.save(plots_dir / f"correlations_{step}.png")

    # Save final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
    }, run_dir / "checkpoint.pt")

    # Save final dashboard
    dashboard = dashboard_figure(
        steps=history["steps"],
        train_losses=history["train_loss"],
        val_losses=history["val_loss"],
        train_accs=history["train_acc"],
        val_accs=history["val_acc"],
        grad_norms=history["grad_norms"],
        grad_cosine=history["grad_cosine"],
    )
    dashboard.save(plots_dir / "dashboard.png")

    return {
        "run_dir": str(run_dir),
        "best_val_acc": best_val_acc,
        "final_step": step,
        "history": history,
    }
