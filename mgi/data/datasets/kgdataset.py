from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Mapping

from pykeen.datasets.base import Dataset, PathDataset
from pykeen.triples import TriplesFactory


class KGDataset:
    def __init__(self, dataset: Dataset, evaluation_restricted_entities: list[int] | None = None):
        self.dataset = dataset
        self.evaluation_restricted_entities = evaluation_restricted_entities

    @classmethod
    def from_path(cls, dataset_path: Path) -> "KGDataset":
        return cls(
            PathDataset(
                dataset_path / "train.tsv",
                dataset_path / "test.tsv",
                dataset_path / "valid.tsv",
            )
        )

    @property
    def training(self) -> TriplesFactory:
        assert isinstance(self.dataset.training, TriplesFactory)
        return self.dataset.training

    @property
    def testing(self) -> TriplesFactory:
        assert isinstance(self.dataset.testing, TriplesFactory)
        return self.dataset.testing

    @property
    def validation(self) -> TriplesFactory:
        assert isinstance(self.dataset.validation, TriplesFactory)
        return self.dataset.validation

    @cached_property
    def id_to_entity(self) -> Mapping[int, str]:
        return self.training.entity_id_to_label

    @cached_property
    def entities_ids(self) -> list[int]:
        return list(sorted(self.id_to_entity.keys()))

    @cached_property
    def entities(self) -> list[str]:
        return [self.id_to_entity[key] for key in self.entities_ids]

    @cached_property
    def anonymized_entities(self) -> list[str]:
        anonymization_map = self.anonymization_map
        return [anonymization_map[e] for e in self.entities]

    @cached_property
    def id_to_anonymized_entity(self) -> dict[int, str]:
        anonymization_map = self.anonymization_map
        assert isinstance(self.dataset.training, TriplesFactory)
        return {
            k: anonymization_map[v] for k, v in self.dataset.training.entity_id_to_label.items()
        }

    @property
    def anonymization_map(self) -> dict[str, str]:
        raise NotImplementedError

    @property
    def inv_anonymization_map(self) -> dict[str, list[str]]:
        i_a_map = defaultdict(list)
        for entity, a_entity in self.anonymization_map.items():
            i_a_map[a_entity].append(entity)
        return dict(i_a_map)

    @cached_property
    def relations(self) -> list[str]:
        return list(sorted(self.dataset.relation_to_id.keys()))
