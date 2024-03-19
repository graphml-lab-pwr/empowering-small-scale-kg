import numpy as np
import numpy.typing as npt

from mgi.data.datasets.kgdataset import KGDataset
from mgi.samplers.base_sampler import BaseSampler


class RandomTripleSampler(BaseSampler):
    def sample(
        self, dataset: KGDataset
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.str_], npt.NDArray[np.str_]]:
        triples = dataset.training.triples
        entities_to_preserve = self._get_nodes(dataset, "validation") | self._get_nodes(
            dataset, "testing"
        )
        relations_to_preserve = self._get_relations(dataset, "validation") | self._get_relations(
            dataset, "testing"
        )

        sampled_triples = []
        sampled_entities = set()
        for e in entities_to_preserve:
            if e in sampled_entities:
                continue
            [sample] = self.random_gen.choice(
                triples[(triples[:, 0] == e) | (triples[:, 2] == e)], 1
            )
            sampled_triples.append(tuple(sample))
            sampled_entities |= set(sample[[0, 2]].tolist())

        sampled_relations = {t[1] for t in sampled_triples}
        for r in relations_to_preserve:
            if r in sampled_relations:
                continue
            [sample] = self.random_gen.choice(triples[triples[:, 1] == r], 1)
            sampled_triples.append(tuple(sample))
            sampled_relations |= {sample[1]}

        to_sample = round(len(triples) * self.p) - len(sampled_triples)
        if to_sample < 0:
            raise ValueError("Impossible to preserve val and test with the given `p`.")

        pool = np.array(list(set(map(tuple, triples)) - set(sampled_triples)))

        sampled_triples += set(map(tuple, self.random_gen.choice(pool, to_sample, replace=False)))
        training_triples = np.array(sampled_triples)

        self.check_if_val_test_preserved(training_triples, dataset)
        return training_triples, dataset.validation.triples, dataset.testing.triples
