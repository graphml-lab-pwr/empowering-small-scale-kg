from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from mgi.training.config import ExperimentConfig

DEFAULT_REPEATS = 3


def _parent_name_impl(_parent_: DictConfig) -> str:
    """Custom resolver for parent key name interpolation.
    Credit: https://github.com/omry/omegaconf/discussions/937#discussioncomment-2787746
    """
    result = _parent_._key()
    assert isinstance(result, str)
    return result


OmegaConf.register_new_resolver("parent_name", _parent_name_impl)


def load_config(
    path: Path, config_key: str | None = None, key_store_name: str | None = None
) -> DictConfig:
    config = OmegaConf.load(path)
    assert isinstance(config, DictConfig)

    if config_key:
        config = config[config_key]
    assert isinstance(config, DictConfig)
    if key_store_name:
        config[key_store_name] = config_key

    return config


def load_training_config(
    gk_dataset_config_key: str | None,
    ds_dataset_config_path: Path,
    ds_dataset_config_key: str,
    model_config_path: Path,
    model_config_key: str,
    gk_dataset_config_path: Path | None = None,
    alignment_config_path: Path | None = None,
    alignment_config_key: str | None = None,
) -> DictConfig:
    if gk_dataset_config_path is not None:
        gk_dataset_config = load_config(
            gk_dataset_config_path, gk_dataset_config_key, "experiment_name"
        )
    else:
        gk_dataset_config = OmegaConf.structured(dict(experiment_name="nogk"))

    ds_dataset_config = load_config(
        ds_dataset_config_path, ds_dataset_config_key, "experiment_name"
    )
    model_config = load_config(model_config_path, model_config_key, "experiment_name")

    if alignment_config_path is not None:
        alignment_config = load_config(
            alignment_config_path, alignment_config_key, "experiment_name"
        )
    else:
        alignment_config = OmegaConf.structured(dict(experiment_name="noalign"))

    experiment_name = "__".join(
        [
            gk_dataset_config.experiment_name,
            ds_dataset_config.experiment_name,
            model_config.experiment_name,
            alignment_config.experiment_name,
        ]
    )

    merged_config = OmegaConf.merge(
        ds_dataset_config, gk_dataset_config, model_config, alignment_config
    )
    merged_config.experiment_name = experiment_name
    merged_config.experiment_dir = Path(merged_config.experiment_dir) / experiment_name.replace(
        "__", "/"
    )

    if "repeats" not in merged_config:  # standard experiment
        assert gk_dataset_config_path is None
        merged_config.repeats = DEFAULT_REPEATS
    elif (
        ds_dataset_config_key.split("_")[0] == gk_dataset_config_key
        and "repeats" in ds_dataset_config
    ):  # FB15K synthetic experiment
        merged_config.repeats = ds_dataset_config.repeats
    else:  # realistic experiment
        merged_config.repeats = gk_dataset_config.repeats

    schema = OmegaConf.structured(ExperimentConfig)
    merged_config = OmegaConf.merge(schema, merged_config)
    assert isinstance(merged_config, DictConfig)
    return merged_config
