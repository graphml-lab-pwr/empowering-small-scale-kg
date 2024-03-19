import pickle
from collections import defaultdict
from copy import copy
from statistics import mean, stdev
from typing import Any

import srsly
import torch
from omegaconf import DictConfig
from pykeen.pipeline import PipelineResult

from mgi.data.datasets.kgdataset import KGDataset


def save(
    pipeline_results: PipelineResult,
    raw_config: DictConfig,
    dataset: KGDataset,
    experiment_num: int,
) -> None:
    metrics = pipeline_results.metric_results.to_dict()  # type: ignore[no-untyped-call]
    saving_dir = raw_config.experiment_dir / "runs" / str(experiment_num)
    saving_dir.mkdir(exist_ok=True, parents=True)
    srsly.write_json(saving_dir / "metrics.json", metrics)
    with open(saving_dir / "model.pt", "wb") as f:
        torch.save(pipeline_results.model, f)
    results = copy(pipeline_results.__dict__)
    for k in ["model", "training", "training_loop", "metric_results", "stopper"]:
        results.pop(k)
    results["configuration"].pop("callbacks")

    with open(saving_dir / "other_results.pkl", "wb") as f:
        pickle.dump(results, f)

    dataset.dataset.to_directory_binary((saving_dir / "dataset").absolute())


def average_metrics(raw_config: DictConfig) -> None:
    metrics = []
    for i in range(raw_config.repeats):
        metrics.append(
            srsly.read_json(raw_config.experiment_dir / "runs" / str(i) / "metrics.json")
        )

    avg_metrics: defaultdict[str, Any] = defaultdict(dict)
    for k1 in metrics[0].keys():
        for k2 in metrics[0][k1].keys():
            averaged = {}
            for metric_name in metrics[0][k1][k2].keys():
                values = list(m[k1][k2][metric_name] for m in metrics)
                averaged[metric_name] = {
                    "mean": mean(values),
                    "std": 0 if len(values) == 1 else stdev(values),
                }
            avg_metrics[k1][k2] = averaged

    srsly.write_json(raw_config.experiment_dir / "avg_metrics.json", avg_metrics)
