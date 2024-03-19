from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Literal

import fasttext
import numpy as np
import numpy.typing as npt
import spacy
from tqdm.auto import tqdm

from mgi.data.datasets.kgdataset import KGDataset
from mgi.defaults import FASTTEXT_MODEL_PATH


class SimilarityEmbedder(ABC):
    @abstractmethod
    def vectorize_dataset(
        self, dataset: KGDataset, subset: Literal["training", "validation", "testing"] | None = None
    ) -> npt.NDArray[np.float32]:
        pass


class SingleTextSimilarityEmbedder(SimilarityEmbedder, ABC):
    def __init__(
        self,
        vectorization: Literal[
            "node", "in_neighborhood", "out_neighborhood", "in_out_neighborhood"
        ],
    ):
        self.vectorization = vectorization

    def vectorize_dataset(
        self,
        dataset: KGDataset,
        subset: Literal["training", "validation", "testing"] | None = None,
    ) -> npt.NDArray[np.float32]:
        if self.vectorization == "node":
            if subset is not None:
                raise ValueError("'Node' vectorization method doesn't consider subsets.")
            return self._vectorize_docs([[e] for e in dataset.anonymized_entities])
        elif self.vectorization in {"in_neighborhood", "out_neighborhood", "in_out_neighborhood"}:
            assert subset in {"training", "validation", "testing"}
            triples = dataset.training.mapped_triples.numpy()
            if subset != "training":
                triples = np.vstack(
                    [triples, dataset.__getattribute__(subset).mapped_triples.numpy()]
                )
                assert len(dataset.training.mapped_triples) < len(triples)
            docs = self._get_docs(triples, np.array(dataset.anonymized_entities))
            return self._vectorize_docs(docs)
        else:
            raise ValueError(f"Wrong vectorization: {self.vectorization}")

    def _get_docs(
        self, triples: npt.NDArray[np.int64], anonymized_entities: npt.NDArray[np.str_]
    ) -> list[list[str]]:
        docs = {}
        for triple in tqdm(triples, desc="Building docs..."):
            s, _, o = triple
            if s not in docs:
                docs[s] = [anonymized_entities[s]]
            if o not in docs:
                docs[o] = [anonymized_entities[o]]
            docs[o].append(anonymized_entities[s])
            docs[s].append(anonymized_entities[o])

        assert len(docs) == max(docs.keys()) + 1

        return [docs[i] for i in sorted(docs)]

    @abstractmethod
    def _vectorize_docs(self, docs: list[list[str]]) -> npt.NDArray[np.float32]:
        pass


class SpacySimilarityEmbedder(SingleTextSimilarityEmbedder):
    def __init__(
        self,
        vectorization: Literal[
            "node", "in_neighborhood", "out_neighborhood", "in_out_neighborhood"
        ],
        spacy_model: str = "en_core_web_lg",
    ):
        super().__init__(vectorization)
        self.nlp = spacy.load(
            spacy_model,
            disable=["tok2vec", "attribute_ruler", "lemmatizer", "ner", "tagger", "parser"],
        )

    def _vectorize_docs(self, docs: list[list[str]], n_jobs: int = -1) -> npt.NDArray[np.float32]:
        joined_docs = [" ".join(doc) for doc in docs]
        return np.array(
            [
                doc.vector
                for doc in tqdm(
                    self.nlp.pipe(joined_docs, n_process=n_jobs),
                    desc="Vectorizing...",
                    total=len(docs),
                )
            ]
        )


class FastTextSimilarityEmbedder(SingleTextSimilarityEmbedder):
    def __init__(
        self,
        vectorization: Literal[
            "node", "in_neighborhood", "out_neighborhood", "in_out_neighborhood"
        ],
        model_path: Path = FASTTEXT_MODEL_PATH,
    ):
        super().__init__(vectorization)
        self.model = fasttext.load_model(str(model_path))

    def _vectorize_docs(self, docs: list[list[str]]) -> npt.NDArray[np.float32]:
        return np.array(
            [
                np.mean([self.model.get_word_vector(word) for word in text], axis=0)
                for text in tqdm(docs, desc="Vectorizing...")
            ]
        )


class MultipleTextSimilarityEmbedder(SimilarityEmbedder, ABC):
    def __init__(
        self,
        vectorization: Literal[
            "node", "in_neighborhood", "out_neighborhood", "in_out_neighborhood"
        ],
    ):
        self.vectorization = vectorization

    def vectorize_dataset(
        self,
        dataset: KGDataset,
        subset: Literal["training", "validation", "testing"] | None = None,
    ) -> npt.NDArray[np.float32]:
        if self.vectorization == "node":
            if subset is not None:
                raise ValueError("'Node' vectorization method doesn't consider subsets.")
            return self._vectorize_docs([(e, [], []) for e in dataset.anonymized_entities])
        elif self.vectorization in {"in_neighborhood", "out_neighborhood", "in_out_neighborhood"}:
            assert subset in {"training", "validation", "testing"}
            triples = dataset.training.mapped_triples.numpy()
            if subset != "training":
                triples = np.vstack(
                    [triples, dataset.__getattribute__(subset).mapped_triples.numpy()]
                )
                assert len(dataset.training.mapped_triples) < len(triples)
            docs = self._get_docs(triples, np.array(dataset.anonymized_entities))
            return self._vectorize_docs(docs)
        else:
            raise ValueError(f"Wrong vectorization: {self.vectorization}")

    def _get_docs(
        self, triples: npt.NDArray[np.int64], anonymized_entities: npt.NDArray[np.str_]
    ) -> list[tuple[str, list[str], list[str]]]:
        self_docs, in_docs, out_docs = dict(), defaultdict(list), defaultdict(list)

        for triple in tqdm(triples, desc="Building docs..."):
            s, _, o = triple
            if s not in self_docs:
                self_docs[s] = anonymized_entities[s]
            if o not in self_docs:
                self_docs[o] = anonymized_entities[o]
            in_docs[o].append(anonymized_entities[s])
            out_docs[s].append(anonymized_entities[o])

        assert len(self_docs) == max(self_docs.keys()) + 1

        return [(self_docs[i], in_docs[i], out_docs[i]) for i in sorted(self_docs)]

    def _vectorize_docs(
        self, docs: list[tuple[str, list[str], list[str]]]
    ) -> npt.NDArray[np.float32]:
        vectors = np.array(
            [
                [
                    self._vectorize_doc([self_doc]),
                    self._vectorize_doc(in_doc),
                    self._vectorize_doc(out_doc),
                ]
                for self_doc, in_doc, out_doc in tqdm(docs, desc="Vectorizing...")
            ]
        )
        return vectors.reshape((-1, 900))

    @abstractmethod
    def _vectorize_doc(self, x: list[str]) -> npt.NDArray[np.float32]:
        pass


class FastTextSimilarityEmbedderLong(MultipleTextSimilarityEmbedder):
    def __init__(
        self,
        vectorization: Literal[
            "node", "in_neighborhood", "out_neighborhood", "in_out_neighborhood"
        ],
        model_path: Path = FASTTEXT_MODEL_PATH,
    ):
        super().__init__(vectorization)
        if self.vectorization in {"node", "in_neighborhood", "out_neighborhood"}:
            raise ValueError(
                f"`{self.vectorization}` vectorization method is not supported in "
                f"FastTextSimilarityEmbedderLong."
            )

        self.model = fasttext.load_model(str(model_path))

    def _vectorize_doc(self, x: list[str]) -> npt.NDArray[np.float32]:
        if len(x) == 0:
            x = [""]
        result = np.mean([self.model.get_word_vector(word) for word in x], axis=0)
        assert isinstance(result, np.ndarray)
        return result
