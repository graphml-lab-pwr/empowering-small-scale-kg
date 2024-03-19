from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import srsly

from mgi.defaults import SAMPLED_DATASETS, SAMPLING_CONFIGS_PATH


@dataclass
class SampledDatasetMetadata:
    name: str
    sampling_config: dict[str, Any]
    path: Path = field(repr=False, init=False)

    def __post_init__(self) -> None:
        self.path = SAMPLED_DATASETS / self.sampling_config["dataset"] / self.name


def load_sampled_datasets_metadata() -> dict[str, SampledDatasetMetadata]:
    result = {}
    for path in SAMPLING_CONFIGS_PATH.glob("*"):
        for name, sampling_config in srsly.read_yaml(path).items():
            if name.count("_") > 1:
                result[name] = SampledDatasetMetadata(name, sampling_config)
    assert len(result) > 0
    return result
