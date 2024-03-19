import numpy as np
import numpy.typing as npt

from mgi.data.datasets.kgdataset import KGDataset
from mgi.samplers.base_sampler import BaseSampler


class RandomNodeSampler(BaseSampler):
    def sample(
        self, dataset: KGDataset
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.str_], npt.NDArray[np.str_]]:
        triples = dataset.training.triples
        val_test_entities = self._get_nodes(dataset, "validation") | self._get_nodes(
            dataset, "testing"
        )
        entities_to_preserve = np.array(list(sorted(val_test_entities)))
        entities, counts = np.unique(triples[:, [0, 2]], return_counts=True)
        entities_degrees: dict[str, int] = dict(zip(entities, counts))

        sampled_entities = set()
        for e in sorted(entities_to_preserve, key=lambda x: entities_degrees[x]):
            if e in sampled_entities:
                continue
            sampled_entities |= self.sample_triple_to_preserve_entity(e, triples)

        relations_to_preserve = self._get_relations(dataset, "validation") | self._get_relations(
            dataset, "testing"
        )
        sampled_relations = set()
        for r in relations_to_preserve:
            if r in sampled_relations:
                continue
            [sample] = self.random_gen.choice(triples[triples[:, 1] == r], 1)
            sampled_entities |= {sample[0], sample[2]}
            sampled_relations |= {sample[1]}

        target_entities_num = round(len(self._get_nodes(dataset, "training")) * self.p)
        to_sample = target_entities_num - len(sampled_entities)
        if to_sample < 0:
            raise ValueError("Failed to preserve val and test with the given `p` and `seed`.")

        entities_pool = np.array(list(self._get_nodes(dataset, "training") - sampled_entities))
        while len(sampled_entities) < target_entities_num:
            [e] = self.random_gen.choice(entities_pool, 1)
            if e in sampled_entities:  # in case of already added entity
                continue
            new_sampled_entities = self.sample_triple_to_preserve_entity(e, triples)
            sampled_entities |= new_sampled_entities

        head_mask = np.in1d(dataset.training.triples[:, 0], list(sampled_entities))
        tail_mask = np.in1d(dataset.training.triples[:, 2], list(sampled_entities))
        mask = head_mask & tail_mask

        training_triples = dataset.training.triples[mask]
        self.check_if_val_test_preserved(training_triples, dataset)
        return training_triples, dataset.validation.triples, dataset.testing.triples

    def sample_triple_to_preserve_entity(
        self, entity: str, triples: npt.NDArray[np.str_]
    ) -> set[np.str_]:
        pool = triples[(triples[:, 0] == entity) | (triples[:, 2] == entity)]
        [sample] = self.random_gen.choice(pool, 1)
        return {sample[0], sample[2]}
