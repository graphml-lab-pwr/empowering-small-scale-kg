from functools import cached_property
from pathlib import Path

from mgi.data.datasets.kgdataset import KGDataset
from mgi.defaults import YAGO310_DATASET_PATH


class YAGO310(KGDataset):
    @classmethod
    def from_path(cls, dataset_path: Path = YAGO310_DATASET_PATH) -> "KGDataset":
        return super().from_path(dataset_path)

    @cached_property
    def anonymization_map(self) -> dict[str, str]:
        entities = list(self.dataset.entity_to_id.keys())
        a_map = {}
        for entity in entities:
            split = entity.split("(")
            a_entity = split[0].replace("_", " ").strip()
            a_map[entity] = a_entity
        return a_map
