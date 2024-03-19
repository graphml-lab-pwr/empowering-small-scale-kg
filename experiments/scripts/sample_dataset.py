from pathlib import Path

import typer

from mgi.data.datasets.dataset_utils import get_gk_dataset
from mgi.data.sampled_datasets import load_sampled_datasets_metadata
from mgi.defaults import SAMPLED_DATASETS
from mgi.samplers.base_sampler import BaseSampler
from mgi.samplers.random_node_sampler import RandomNodeSampler
from mgi.samplers.random_relation_sampler import RandomRelationSampler
from mgi.samplers.random_triple_sampler import RandomTripleSampler


def main(
    ds_dataset_name: str = typer.Argument(..., help="Dataset name"),
    seeds: list[int] = typer.Option(..., help="Seeds"),
) -> None:
    ds_dataset_metadatas = load_sampled_datasets_metadata()
    config = ds_dataset_metadatas[ds_dataset_name].sampling_config
    for seed in seeds:
        kg_dataset = get_gk_dataset(config["dataset"])
        if config["sampling"] == "node":
            sampler: BaseSampler = RandomNodeSampler(config["p"], seed=seed)
        elif config["sampling"] == "triple":
            sampler = RandomTripleSampler(config["p"], seed=seed)
        elif config["sampling"] == "relation":
            sampler = RandomRelationSampler(config["p"], seed=seed)
        else:
            raise ValueError()

        output_path = SAMPLED_DATASETS / ds_dataset_name / str(seed)
        output_path.mkdir(exist_ok=True, parents=True)
        sampler.sample_and_save(kg_dataset, output_path)


typer.run(main)
