from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from mgi.data.datasets.kgdataset import KGDataset


class BaseSampler(ABC):
    def __init__(self, p: float, seed: int = 17):
        """
        :param p: probability of sampling
        """
        self.p = p
        self.random_gen = np.random.default_rng(seed)

    def sample_and_save(self, dataset: KGDataset, output_path: Path) -> None:
        training, _, _ = self.sample(dataset)
        self.save_triples(training, output_path, "training")

        for subset in ["validation", "testing"]:
            triples = dataset.__getattribute__(subset).triples.tolist()
            self.save_triples(triples, output_path, subset)

    @abstractmethod
    def sample(
        self, dataset: KGDataset
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.str_], npt.NDArray[np.str_]]:
        pass

    def save_triples(
        self,
        triples: npt.NDArray[np.str_],
        output_path: Path,
        subset: str,
    ) -> None:
        subset_file_map = {
            "testing": "test.tsv",
            "training": "train.tsv",
            "validation": "valid.tsv",
        }

        pd.DataFrame(triples).to_csv(
            output_path / subset_file_map[subset], index=False, header=False, sep="\t"
        )

    def _get_nodes(self, dataset: KGDataset, subset: str) -> set[str]:
        return set(getattr(dataset, subset).triples[:, [0, 2]].flatten())

    def _get_relations(self, dataset: KGDataset, subset: str) -> set[str]:
        return set(getattr(dataset, subset).triples[:, 1].flatten())

    def check_if_val_test_preserved(
        self, training_triples: npt.NDArray[np.str_], dataset: KGDataset
    ) -> None:
        training_entities = set(training_triples[:, [0, 2]].flatten())
        training_relations = set(training_triples[:, 1])

        for subset in ["validation", "testing"]:
            triples = dataset.__getattribute__(subset).triples
            for h, r, t in triples:
                assert h in training_entities
                assert t in training_entities
                assert r in training_relations
