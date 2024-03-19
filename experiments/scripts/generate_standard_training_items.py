import itertools
from pathlib import Path

import srsly
import typer
from rich import print

from mgi.defaults import ROOT_PATH, TRAINING_CONFIGS_PATH

GK_DATASETS_CONFIG = TRAINING_CONFIGS_PATH / "gk_datasets.yaml"
MODEL_CONFIG = TRAINING_CONFIGS_PATH / "model.yaml"


def main(
    generation_config_path: Path = typer.Option(
        TRAINING_CONFIGS_PATH / "generation_config_standard_training_items.yaml", exists=True
    ),
    generation_config_key: str = typer.Option(...),
    output_path: Path = typer.Option(...),
) -> None:
    print(locals())
    generation_config = srsly.read_yaml(generation_config_path)[generation_config_key]
    gk_datasets = generation_config["gk_datasets"]
    models = generation_config["models"]
    ds_datasets = generation_config["ds_datasets"]

    all_items = []

    for gk_dataset, ds_dataset, model in itertools.product(
        sorted(gk_datasets), sorted(ds_datasets), sorted(models)
    ):
        all_items.append(
            {
                "gk_dataset_config_key": gk_dataset,  # only for defining ds dep in dvc.yaml
                "ds_dataset_config_path": generation_config["ds_datasets_config_path"],
                "ds_dataset_config_key": ds_dataset,
                "model_config_path": str(MODEL_CONFIG.relative_to(ROOT_PATH)),
                "model_config_key": model,
            }
        )

    srsly.write_yaml(output_path, {f"standard_{generation_config_key}_training_items": all_items})


typer.run(main)
