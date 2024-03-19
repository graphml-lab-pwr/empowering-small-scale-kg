import os
from pathlib import Path

import torch
import typer
from dotenv import load_dotenv
from lightning_lite import seed_everything
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print
from tqdm import tqdm

from mgi.data.datasets.dataset_utils import get_ds_dataset
from mgi.defaults import ROOT_PATH
from mgi.training.config import TrainingConfig
from mgi.training.training_pipeline import alter_config_when_debug, train_model
from mgi.training.utils import save
from mgi.utils.config import load_training_config


def main(
    ds_dataset_config_path: Path = typer.Option(..., exists=True),
    ds_dataset_config_key: str = typer.Option(...),
    model_config_path: Path = typer.Option(..., exists=True),
    model_config_key: str = typer.Option(...),
) -> None:
    load_dotenv(ROOT_PATH / ".env")  # Load environment variables
    if TORCH_N_THREADS := os.getenv("TORCH_N_THREADS"):
        torch.set_num_threads(int(TORCH_N_THREADS))
    raw_config = load_training_config(
        gk_dataset_config_path=None,
        gk_dataset_config_key=None,
        ds_dataset_config_path=ds_dataset_config_path,
        ds_dataset_config_key=ds_dataset_config_key,
        model_config_path=model_config_path,
        model_config_key=model_config_key,
        alignment_config_path=None,
        alignment_config_key=None,
    )
    raw_config.experiment_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(config=raw_config, f=raw_config.experiment_dir / f"config.yaml")
    assert isinstance(raw_config.random_seed, ListConfig | list)
    assert len(raw_config.random_seed) >= raw_config.repeats
    for i, random_seed in tqdm(
        enumerate(raw_config.random_seed[: raw_config.repeats]), desc="Experiment repeat"
    ):
        experiment_raw_config = OmegaConf.create(OmegaConf.to_container(raw_config))
        experiment_raw_config.random_seed = random_seed
        experiment_raw_config = OmegaConf.merge(
            OmegaConf.structured(TrainingConfig), experiment_raw_config
        )
        experiment_raw_config.experiment_name += f"_{random_seed}"

        seed_everything(experiment_raw_config.random_seed)
        assert isinstance(experiment_raw_config, DictConfig)
        experiment_raw_config = alter_config_when_debug(experiment_raw_config)
        print(OmegaConf.to_container(experiment_raw_config, resolve=True))
        dataset = get_ds_dataset(ds_dataset_name=experiment_raw_config.ds_dataset, seed=random_seed)
        print(dataset)

        pipeline_results = train_model(dataset, experiment_raw_config)
        save(pipeline_results, experiment_raw_config, dataset, i)


typer.run(main)
