from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class _BaseConfig(ABC):
    experiment_name: str
    gk_dataset: str
    ds_dataset: str
    mapping_metric: str
    vectorization: str
    alignment_k: int
    combine_method: str
    crop_gk_n: int | None

    model: str
    model_kwargs: dict[str, Any]
    weight_method: str | None
    loss: str
    loss_kwargs: dict[str, Any]
    optimizer: str
    optimizer_kwargs: dict[str, Any]
    lr_scheduler: bool | None
    lr_scheduler_kwargs: dict[str, Any] | None
    negative_sampler: str
    negative_sampler_kwargs: dict[str, Any]
    automatic_memory_optimization: bool
    training_kwargs: dict[str, Any]
    training_loop: str
    training_loop_kwargs: dict[str, Any] | None

    repeats: int
    evaluation_batch_size: int
    evaluation_frequency: int
    experiment_dir: Path
    wandb_entity: str
    wandb_project: str


@dataclass
class ExperimentConfig(_BaseConfig):
    """Configuration for experiment that consists of multiple runs with different random seeds."""

    random_seed: list[int]


@dataclass
class TrainingConfig(_BaseConfig):
    """Configuration for a single training run."""

    random_seed: int
