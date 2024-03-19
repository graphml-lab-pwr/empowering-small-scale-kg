import warnings
from typing import Any, NamedTuple, Optional

import torch
from pykeen.triples import TriplesFactory
from pykeen.triples.instances import BatchedSLCWAInstances
from torch.utils.data import Dataset


class WeightedSLCWABatch(NamedTuple):
    positives: torch.LongTensor
    negatives: torch.LongTensor
    masks: Optional[torch.BoolTensor]
    weights: torch.Tensor


class WeightedBatchedSLCWAInstances(BatchedSLCWAInstances):
    def __init__(self, weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.weights = weights

    def __getitem__(self, item: list[int]) -> WeightedSLCWABatch:  # type: ignore
        batch = super().__getitem__(item)
        return WeightedSLCWABatch(
            positives=batch.positives,
            negatives=batch.negatives,
            masks=batch.masks,
            weights=self.weights[item],
        )


class WeightedTriplesFactory(TriplesFactory):
    def __init__(self, triples_factory: TriplesFactory, weights: torch.Tensor) -> None:
        assert isinstance(triples_factory.mapped_triples, torch.LongTensor)
        super().__init__(
            mapped_triples=triples_factory.mapped_triples,
            entity_to_id=triples_factory.entity_to_id,
            relation_to_id=triples_factory.relation_to_id,
            create_inverse_triples=triples_factory.create_inverse_triples,
            metadata=triples_factory.metadata,
            num_entities=triples_factory.num_entities,
            num_relations=triples_factory.num_relations,
        )
        self.weights = weights

    def create_slcwa_instances(
        self, *, sampler: Optional[str] = None, **kwargs: Any
    ) -> WeightedBatchedSLCWAInstances:
        # Function is a copy of the original pykeen function with the only difference that the
        # weights are used.
        """Create sLCWA instances for this factory's triples."""
        if sampler is not None:
            raise NotImplementedError
        cls = WeightedBatchedSLCWAInstances
        if "shuffle" in kwargs:
            if kwargs.pop("shuffle"):
                warnings.warn("Training instances are always shuffled.", DeprecationWarning)
            else:
                raise AssertionError("If shuffle is provided, it must be True.")
        assert isinstance(self.mapped_triples, torch.LongTensor)
        return cls(
            weights=self.weights,
            mapped_triples=self._add_inverse_triples_if_necessary(
                mapped_triples=self.mapped_triples
            ),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            **kwargs,
        )

    def create_lcwa_instances(
        self, use_tqdm: Optional[bool] = None, target: Optional[int] = None
    ) -> Dataset[WeightedSLCWABatch]:
        raise NotImplementedError
