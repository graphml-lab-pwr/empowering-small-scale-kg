import os
import time
from functools import cached_property
from typing import Literal

import faiss
import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

from mgi.data.datasets.kgdataset import KGDataset
from mgi.mappings.base_mapping import BaseMapping
from mgi.mappings.similarity_embeddings import SimilarityEmbedder


class NNMapping(BaseMapping):
    def __init__(
        self,
        dataset: KGDataset,
        n: int,
        similarity_embedder: SimilarityEmbedder,
        subset: Literal["training", "validation", "testing"] | None = None,
        metric: str = "cosine",
    ):
        self.dataset = dataset
        self.n = n
        self.similarity_embedder = similarity_embedder
        self.subset = subset

        self.model = NearestNeighbors(
            n_neighbors=n,
            metric=metric,
            algorithm="auto",
            n_jobs=int(os.getenv("SKLEARN_N_JOBS", -1)),
        )
        self.model.fit(self.vectorized_dataset)

    @cached_property
    def vectorized_dataset(self) -> npt.NDArray[np.float32]:
        return self.similarity_embedder.vectorize_dataset(self.dataset, self.subset)

    @property
    def node_map(self) -> dict[str, list[tuple[str, float]]]:
        raise NotImplementedError

    def get_neighbours_map_from_ids(
        self, entities_ids: list[int]
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]:
        x = [self.vectorized_dataset[e_i] for e_i in entities_ids]
        dists, neighbours_ids = self.model.kneighbors(x, self.n, return_distance=True)
        return neighbours_ids, dists

    def get_neighbours_map_from_dataset(
        self, dataset: KGDataset
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]:
        x = self.similarity_embedder.vectorize_dataset(dataset, self.subset)
        print("\nFinding neighbors...")
        start_time = time.time()
        dists, neighbours_ids = self.model.kneighbors(x, self.n, return_distance=True)
        end_time = time.time()
        print("Found neighbors in:", (end_time - start_time) / 60, " min")
        return neighbours_ids, dists


class FaissNNMapping(BaseMapping):
    def __init__(
        self,
        dataset: KGDataset,
        n: int,
        similarity_embedder: SimilarityEmbedder,
        subset: Literal["training", "validation", "testing"] | None = None,
        metric: str = "cosine",
    ):
        assert metric == "euclidean"
        self.dataset = dataset
        self.n = n
        self.similarity_embedder = similarity_embedder
        self.subset = subset

        vectorized_dataset = self.vectorized_dataset
        self.index = faiss.IndexFlatL2(vectorized_dataset.shape[1])
        self.index.add(vectorized_dataset)

    @cached_property
    def vectorized_dataset(self) -> npt.NDArray[np.float32]:
        return self.similarity_embedder.vectorize_dataset(self.dataset, self.subset)

    @property
    def node_map(self) -> dict[str, list[tuple[str, float]]]:
        raise NotImplementedError

    def get_neighbours_map_from_ids(
        self, entities_ids: list[int]
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]:
        x = [self.vectorized_dataset[e_i] for e_i in entities_ids]
        neighbours_ids, dists = self.index.search(x, self.n)
        return neighbours_ids, dists

    def get_neighbours_map_from_dataset(
        self, dataset: KGDataset
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]:
        x = self.similarity_embedder.vectorize_dataset(dataset, self.subset)
        print("\nFinding neighbors...")
        start_time = time.time()
        dists, neighbours_ids = self.index.search(x, self.n)
        end_time = time.time()
        print("Found neighbors in:", (end_time - start_time) / 60, " min")
        return neighbours_ids, dists
