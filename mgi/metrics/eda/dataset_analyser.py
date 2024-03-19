import itertools
from statistics import mean, median
from typing import Any

import networkx as nx
from nested_dict import nested_dict

from mgi.data.datasets.kgdataset import KGDataset


class DatasetAnalyser:
    def __init__(self, dataset: KGDataset):
        self.dataset = dataset

    def entities(self, subset: str | None = None) -> set[str | int]:
        if subset is None:
            return (
                self.entities("training")
                .union(self.entities("validation"))
                .union(self.entities("testing"))
            )
        else:
            return set(
                itertools.chain.from_iterable(
                    (h, t) for h, _, t in self.dataset.dataset.__getattribute__(subset).triples
                )
            )

    def num_entities(self, subset: str | None = None) -> int:
        return len(self.entities(subset))

    def get_metrics(self) -> dict[Any, Any]:
        metrics = nested_dict()
        for subset in ["training", "validation", "testing"]:
            metrics[subset]["num_entities"] = self.num_entities("training")
        metrics["all"]["num_entities"] = self.num_entities()

        for subset in ["validation", "testing"]:
            metrics[subset]["new_entities"] = len(self.entities(subset) - self.entities("training"))

        g = nx.Graph()
        assert hasattr(self.dataset.dataset.training, "triples")
        g.add_edges_from([(h, t) for h, _, t in self.dataset.dataset.training.triples])

        metrics["training"]["n_connected_components"] = nx.number_connected_components(g)
        metrics["training"]["mean_size_of_connected_components"] = mean(
            len(c) for c in nx.connected_components(g)
        )
        metrics["training"]["median_size_of_connected_components"] = median(
            len(c) for c in nx.connected_components(g)
        )

        for subset in ["validation", "testing"]:
            g_copy = g.copy()
            g_copy.add_edges_from(
                (h, t) for h, _, t in self.dataset.dataset.__getattribute__(subset).triples
            )

            metrics[subset]["n_connected_components"] = nx.number_connected_components(g_copy)
            metrics[subset]["mean_size_of_connected_components"] = mean(
                len(c) for c in nx.connected_components(g_copy)
            )
            metrics[subset]["median_size_of_connected_components"] = median(
                len(c) for c in nx.connected_components(g_copy)
            )

        result = metrics.to_dict()
        assert isinstance(result, dict)
        return result
