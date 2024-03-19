from pathlib import Path

import numpy as np
import numpy.typing as npt

from mgi.data.datasets.kgdataset import KGDataset
from mgi.samplers.base_sampler import BaseSampler


class RandomRelationSampler(BaseSampler):
    def sample_and_save(self, dataset: KGDataset, output_path: Path) -> None:
        relations = dataset.relations
        self.random_gen.shuffle(relations)
        sampled_relations = relations[: round(len(relations) * self.p)]
        for subset in ["training", "validation", "testing"]:
            triples = dataset.__getattribute__(subset).triples
            mask = np.in1d(triples[:, 1], sampled_relations)
            sampled_triples = triples[mask]
            self.save_triples(sampled_triples, output_path, subset)

    def sample(
        self, dataset: KGDataset
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.str_], npt.NDArray[np.str_]]:
        relations = dataset.relations
        self.random_gen.shuffle(relations)
        sampled_relations = relations[: round(len(relations) * self.p)]
        return (
            self._sample_subset(dataset, "training", sampled_relations),
            self._sample_subset(dataset, "validation", sampled_relations),
            self._sample_subset(dataset, "testing", sampled_relations),
        )

    def _sample_subset(
        self, dataset: KGDataset, subset: str, sampled_relations: list[str]
    ) -> npt.NDArray[np.str_]:
        triples = dataset.__getattribute__(subset).triples
        mask = np.in1d(triples[:, 1], sampled_relations)
        sampled_triples: npt.NDArray[np.str_] = triples[mask]
        return sampled_triples
