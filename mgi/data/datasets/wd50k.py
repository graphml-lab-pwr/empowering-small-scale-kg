from functools import cached_property
from pathlib import Path

import srsly
from pykeen.datasets import Dataset, PathDataset

from mgi.data.datasets.kgdataset import KGDataset
from mgi.defaults import WD50K_DATASET_PATH


class WD50K(KGDataset):
    def __init__(
        self,
        dataset: Dataset,
        evaluation_restricted_entities: list[int] | None = None,
        entityid_to_entity: dict[str, str] | None = None,
    ) -> None:
        super().__init__(dataset, evaluation_restricted_entities)
        if entityid_to_entity is None:
            entityid_to_entity = self.load_default_entityid_to_entity()
        self.entityid_to_entity = entityid_to_entity

    @classmethod
    def from_path(cls, dataset_path: Path = WD50K_DATASET_PATH) -> "KGDataset":
        entityid_to_entity = cls.load_default_entityid_to_entity()
        return cls(
            PathDataset(
                dataset_path / "train.tsv",
                dataset_path / "test.tsv",
                dataset_path / "valid.tsv",
            ),
            entityid_to_entity=entityid_to_entity,
        )

    @cached_property
    def anonymization_map(self) -> dict[str, str]:
        entities = list(self.dataset.entity_to_id.keys())
        a_map = {}
        for entity in entities:
            a_map[entity] = self.entityid_to_entity.get(entity, entity)
        return a_map

    @classmethod
    def load_default_entityid_to_entity(cls) -> dict[str, str]:
        result = srsly.read_json(WD50K_DATASET_PATH / "qid_to_label.json")
        assert isinstance(result, dict)
        return result
