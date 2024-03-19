from functools import cached_property
from pathlib import Path

from mgi.data.datasets.kgdataset import KGDataset
from mgi.defaults import CONCEPTNET_EN_DATASET_PATH


class ConceptNet(KGDataset):
    @classmethod
    def from_path(cls, dataset_path: Path = CONCEPTNET_EN_DATASET_PATH) -> "KGDataset":
        return super().from_path(dataset_path)

    @cached_property
    def anonymization_map(self) -> dict[str, str]:
        entities = list(self.dataset.entity_to_id.keys())
        a_map = {}
        for entity in entities:
            split = entity.split("/")
            a_map[entity] = split[3].replace("_", " ")
        return a_map
