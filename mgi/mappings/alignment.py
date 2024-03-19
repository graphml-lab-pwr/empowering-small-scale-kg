from abc import ABC, abstractmethod
from typing import Literal, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from mpire import WorkerPool
from pykeen.datasets import EagerDataset
from pykeen.datasets.ea.combination import ExtraRelationGraphPairCombinator, GraphPairCombinator
from pykeen.triples import TriplesFactory
from tqdm.auto import tqdm

from mgi.data.datasets.kgdataset import KGDataset
from mgi.mappings.base_mapping import BaseMapping


class Alignment(ABC):
    @abstractmethod
    def get_alignment(self, ds_dataset: KGDataset, mapping: BaseMapping) -> pd.DataFrame:
        pass


class KNearestAlignment(Alignment):
    def __init__(self, k: int) -> None:
        self.k = k

    def get_alignment(self, ds_dataset: KGDataset, mapping: BaseMapping) -> pd.DataFrame:
        neighbours, dists = mapping.get_neighbours_map_from_dataset(ds_dataset)
        alignment = pd.DataFrame(
            {
                "left": list(range(len(neighbours))),
                "right": neighbours[:, : self.k].tolist(),
                "dist": dists[:, : self.k].tolist(),
            }
        )
        alignment = alignment.explode(["right", "dist"], ignore_index=True)
        alignment = alignment.astype({"left": "int", "right": "int"})
        return alignment


class BaseCombinator(ABC):
    @abstractmethod
    def combine(
        self,
        ds_dataset: KGDataset,
        gk_dataset: KGDataset,
        alignment: Alignment,
        mapping: BaseMapping,
    ) -> tuple[KGDataset, pd.DataFrame, TriplesFactory]:
        pass


class Combinator(BaseCombinator):
    def __init__(self, method: Literal["extra_relation"]):
        self.combinator: GraphPairCombinator
        self.method = method
        if method == "extra_relation":
            self.combinator = ExtraRelationGraphPairCombinator()
        else:
            raise ValueError()

    def combine(
        self,
        ds_dataset: KGDataset,
        gk_dataset: KGDataset,
        alignment: Alignment,
        mapping: BaseMapping,
        crop_gk_n: int | None = None,
        n_jobs: int | None = None,
    ) -> tuple[KGDataset, pd.DataFrame, TriplesFactory]:
        alignment_df = alignment.get_alignment(ds_dataset, mapping)
        if crop_gk_n is not None:
            assert crop_gk_n > 0
            gk_triples_factory, alignment_df = self.crop_gk(
                gk_dataset.training, alignment_df, crop_gk_n
            )
        else:
            gk_triples_factory = gk_dataset.training
        training, _ = self.combinator(
            ds_dataset.training, gk_triples_factory, alignment_df.drop(columns=["dist"])
        )

        new_relation_ids_map = {
            r: self._find_new_id(training.relation_to_id, r)
            for r in tqdm(ds_dataset.relations, desc="Relation mapping...")
        }

        with WorkerPool(n_jobs=n_jobs, shared_objects=training.entity_to_id) as pool:
            new_entity_id_map = dict(
                zip(
                    ds_dataset.entities,
                    pool.map(
                        self._find_new_id,
                        ds_dataset.entities,
                        progress_bar=True,
                        progress_bar_options={"desc": "Entity mapping..."},
                    ),
                )
            )

        validation = TriplesFactory(
            mapped_triples=self._map_triples(
                ds_dataset.validation.triples, new_entity_id_map, new_relation_ids_map
            ),
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id,
        )
        testing = TriplesFactory(
            mapped_triples=self._map_triples(
                ds_dataset.testing.triples, new_entity_id_map, new_relation_ids_map
            ),
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id,
        )

        assert ds_dataset.validation.num_triples == validation.num_triples
        assert ds_dataset.testing.num_triples == testing.num_triples

        self.check_combination(ds_dataset, training, testing, validation)

        combined_dataset = KGDataset(
            dataset=EagerDataset(
                training,
                testing,
                validation,
                metadata=ds_dataset.dataset.metadata,
            ),
            evaluation_restricted_entities=self.get_evaluation_restricted_entities(
                training, ds_dataset
            ),
        )

        return combined_dataset, alignment_df, gk_triples_factory

    def crop_gk(
        self, gk_triples: TriplesFactory, alignment: pd.DataFrame, n: int
    ) -> tuple[TriplesFactory, pd.DataFrame]:
        triples = gk_triples.mapped_triples.numpy()
        n_neighborhood = list(alignment.right.unique())
        for n in range(n):
            mask = np.isin(triples[:, 0], n_neighborhood) | np.isin(triples[:, 2], n_neighborhood)
            n_neighborhood = list(set(triples[mask][:, [0, 2]].flatten()))
        cropped_triples = gk_triples.triples[
            np.isin(triples[:, 0], n_neighborhood) & np.isin(triples[:, 2], n_neighborhood)
        ]
        cropped_triples_factory = TriplesFactory.from_labeled_triples(cropped_triples)

        cropped_alignment = alignment.copy()

        right_ent = cropped_alignment.right.map(gk_triples.entity_id_to_label)
        cropped_alignment["right"] = right_ent.map(cropped_triples_factory.entity_to_id)
        return cropped_triples_factory, cropped_alignment

    def check_combination(
        self,
        ds_dataset: KGDataset,
        training: TriplesFactory,
        testing: TriplesFactory,
        validation: TriplesFactory,
    ) -> None:
        for k in training.entity_id_to_label.keys():
            assert (
                training.entity_id_to_label[k]
                == validation.entity_id_to_label[k]
                == testing.entity_id_to_label[k]
            )

        for mapped_entity, org_entity in zip(
            validation.triples[:, 0], ds_dataset.validation.triples[:, 0]
        ):
            assert org_entity in mapped_entity
        for mapped_r, org_r in zip(validation.triples[:, 1], ds_dataset.validation.triples[:, 1]):
            assert org_r in mapped_r
        for mapped_entity, org_entity in zip(
            validation.triples[:, 2], ds_dataset.validation.triples[:, 2]
        ):
            assert org_entity in mapped_entity
        for mapped_entity, org_entity in zip(
            testing.triples[:, 0], ds_dataset.testing.triples[:, 0]
        ):
            assert org_entity in mapped_entity
        for mapped_r, org_r in zip(testing.triples[:, 1], ds_dataset.testing.triples[:, 1]):
            assert org_r in mapped_r
        for mapped_entity, org_entity in zip(
            testing.triples[:, 2], ds_dataset.testing.triples[:, 2]
        ):
            assert org_entity in mapped_entity

    def _map_triples(
        self,
        triples: npt.NDArray[np.int_],
        new_entity_id_map: Mapping[str, int],
        new_relation_ids_map: Mapping[str, int],
    ) -> torch.LongTensor:
        array = [
            [
                new_entity_id_map[h],
                new_relation_ids_map[r],
                new_entity_id_map[t],
            ]
            for h, r, t in triples
        ]
        return torch.LongTensor(array)

    def _find_new_id(self, mapping: Mapping[str, int], org_name: str) -> int:
        mapped_name = f"left:{org_name}"
        return mapping[mapped_name]

    def get_evaluation_restricted_entities(
        self, training: TriplesFactory, ds_dataset: KGDataset
    ) -> list[int]:
        evaluation_restricted_entities = [
            e for e in training.entity_id_to_label if "left:" in training.entity_id_to_label[e]
        ]
        if self.method == "extra_relation":
            assert len(evaluation_restricted_entities) == len(ds_dataset.entities)
        else:
            raise ValueError()

        return evaluation_restricted_entities
