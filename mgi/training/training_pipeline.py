import os
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pykeen.evaluation import RankBasedEvaluator
from pykeen.metrics.ranking import (
    ArithmeticMeanRank,
    HitsAtK,
    InverseHarmonicMeanRank,
    RankBasedMetric,
)
from pykeen.pipeline import PipelineResult, pipeline
from pykeen.trackers import WANDBResultTracker
from pykeen.training.callbacks import EvaluationLoopTrainingCallback, TrainingCallback
from torch import LongTensor

from mgi.data.datasets.kgdataset import KGDataset
from mgi.loss import get_loss
from mgi.training import get_training_loop

# in case of unstable connection use the WANDB_MODE=offline env variable
# in case of debugging, one should change WANDB project or use offline and don't sync
WANDB_DEBUG_PROJECT = os.getenv("WANDB_DEBUG_PROJECT", None)
# in case of debugging decrease workers to 0 (avoid PyCharm hangs)
DEBUG_MODE = bool(os.getenv("DEBUG_MODE", False))


def train_model(dataset: KGDataset, raw_config: DictConfig) -> PipelineResult:
    result_tracker = WANDBResultTracker(
        entity=raw_config.wandb_entity,
        project=raw_config.wandb_project,
        dir=raw_config.experiment_dir,
    )
    result_tracker.start_run(run_name=raw_config.experiment_name)
    config = convert_dictconfig_to_dict(raw_config)
    config["training_kwargs"]["callbacks"] = get_callbacks(dataset, config)
    result_tracker.log_params({"config": config})

    assert isinstance(dataset.testing.mapped_triples, LongTensor)
    evaluation_kwargs = dict(
        batch_size=raw_config.evaluation_batch_size,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
        restrict_entities_to=dataset.evaluation_restricted_entities,
    )
    training_loop = get_training_loop(config["training_loop"])
    results = pipeline(
        dataset=dataset.dataset,
        model=config["model"],
        model_kwargs=config["model_kwargs"],
        loss=get_loss(config["loss"]),
        loss_kwargs=config["loss_kwargs"],
        optimizer=config["optimizer"],
        optimizer_kwargs=config["optimizer_kwargs"],
        lr_scheduler=config["lr_scheduler"],
        lr_scheduler_kwargs=config["lr_scheduler_kwargs"],
        training_loop=training_loop,
        training_loop_kwargs=None,
        negative_sampler=config["negative_sampler"],
        negative_sampler_kwargs=config["negative_sampler_kwargs"],
        training_kwargs=config["training_kwargs"],
        evaluator=RankBasedEvaluator,
        evaluation_kwargs=evaluation_kwargs,
        result_tracker=result_tracker,
        evaluation_fallback=True,
    )
    return results


def convert_dictconfig_to_dict(config: DictConfig) -> dict[Any, Any]:
    result = OmegaConf.to_container(config, resolve=True)
    assert isinstance(result, dict)
    return result


def alter_config_when_debug(raw_config: DictConfig) -> DictConfig:
    if DEBUG_MODE:
        raw_config["training_kwargs"]["num_workers"] = 0

        if raw_config.get("wandb_project"):
            assert (
                WANDB_DEBUG_PROJECT is not None
                and WANDB_DEBUG_PROJECT != raw_config["wandb_project"]
            ), "When running in debug mode, ensure wandb project other than in config"
            raw_config["wandb_project"] = WANDB_DEBUG_PROJECT
    return raw_config


def get_callbacks(dataset: KGDataset, config: dict[str, Any]) -> list[TrainingCallback]:
    """
    Keep in mind that the metric values calculated during training can differ from those calculated
    after training is completed.  This is due to the fact that evaluation during training
    is done without restricting entities to DKG.
    """
    assert isinstance(dataset.training.mapped_triples, LongTensor)
    assert isinstance(dataset.testing.mapped_triples, LongTensor)
    assert isinstance(dataset.validation.mapped_triples, LongTensor)
    callbacks: list[TrainingCallback] = [
        EvaluationLoopTrainingCallback(
            dataset.validation,
            prefix="val/",
            evaluator=RankBasedEvaluator,
            evaluator_kwargs={"metrics": get_callbacks_metrics(), "add_defaults": False},
            additional_filter_triples=[
                dataset.training.mapped_triples,
                dataset.testing.mapped_triples,
            ],
            batch_size=config["evaluation_batch_size"],
            frequency=config["evaluation_frequency"],
        ),
        EvaluationLoopTrainingCallback(
            dataset.testing,
            prefix="test/",
            evaluator=RankBasedEvaluator,
            evaluator_kwargs={"metrics": get_callbacks_metrics(), "add_defaults": False},
            additional_filter_triples=[
                dataset.validation.mapped_triples,
                dataset.training.mapped_triples,
            ],
            batch_size=config["evaluation_batch_size"],
            frequency=config["evaluation_frequency"],
        ),
    ]
    return callbacks


def get_callbacks_metrics() -> list[RankBasedMetric]:
    return [
        HitsAtK(k=1),
        HitsAtK(k=3),
        HitsAtK(k=10),
        ArithmeticMeanRank(),
        InverseHarmonicMeanRank(),
    ]
