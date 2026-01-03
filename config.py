from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import yaml


@dataclass
class TaskConfig:
    n: int = 10
    k: int = 5
    train_fraction: float = 0.8
    seed: int = 42


@dataclass
class ModelConfig:
    widths: list[int] = field(default_factory=lambda: [64, 64])
    activation: Literal["relu", "tanh", "sigmoid"] = "relu"


@dataclass
class OptimizerConfig:
    type: Literal["adam", "cmaes"] = "adam"
    lr: float = 0.001
    # CMA-ES
    sigma0: float = 1.0


@dataclass
class NoiseConfig:
    enabled: bool = False
    std: float = 0.1
    K: int = 1  # number of samples for gradient averaging


@dataclass
class TrainingConfig:
    max_steps: int = 10000
    eval_every: int = 100
    patience: int = 1000
    batch_size: int | None = None  # None = full batch


@dataclass
class LoggingConfig:
    run_name: str = "experiment"
    log_correlations_every: int = 500


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> Config:
    """Load config from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return Config(
        task=TaskConfig(**data.get("task", {})),
        model=ModelConfig(**data.get("model", {})),
        optimizer=OptimizerConfig(**data.get("optimizer", {})),
        noise=NoiseConfig(**data.get("noise", {})),
        training=TrainingConfig(**data.get("training", {})),
        logging=LoggingConfig(**data.get("logging", {})),
    )


def save_config(config: Config, path: str | Path) -> None:
    """Save config to YAML file."""
    from dataclasses import asdict
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
