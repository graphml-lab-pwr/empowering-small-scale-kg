from typing import Literal

import pandas as pd
import torch
from pykeen.datasets import EagerDataset
from pykeen.datasets.ea.combination import ExtraRelationGraphPairCombinator
from pykeen.triples import TriplesFactory

from mgi.data.datasets.dataset_utils import get_ds_dataset, get_gk_dataset
from mgi.data.datasets.kgdataset import KGDataset
from mgi.data.weighted import WeightedTriplesFactory
from mgi.mappings.alignment import Combinator, KNearestAlignment
from mgi.mappings.nn_mapping import FaissNNMapping
from mgi.mappings.similarity_embeddings import FastTextSimilarityEmbedderLong


def get_combined_dataset(
    ds_dataset_name: str,
    gk_dataset_name: str,
    alignment_k: int,
    vectorization: Literal["node", "in_neighborhood", "out_neighborhood", "in_out_neighborhood"],
    mapping_metric: str,
    combine_method: Literal["extra_relation"],
    seed: int,
    weight_method: Literal["similarity"] | None = None,
    crop_gk_n: int | None = None,
) -> KGDataset:
    gk_dataset = get_gk_dataset(gk_dataset_name)
    ds_dataset = get_ds_dataset(ds_dataset_name, seed)

    alignment = KNearestAlignment(alignment_k)
    embedder = FastTextSimilarityEmbedderLong(vectorization)
    mapping = FaissNNMapping(gk_dataset, alignment_k, embedder, "training", mapping_metric)

    combined_dataset, alignment_df, gk_triples_factory = Combinator(combine_method).combine(
        ds_dataset, gk_dataset, alignment, mapping, crop_gk_n
    )

    if weight_method is not None:
        combined_dataset = add_weights_to_dataset(
            ds_dataset, gk_triples_factory, combined_dataset, alignment_df, weight_method
        )

    return combined_dataset


def add_weights_to_dataset(
    ds_dataset: KGDataset,
    gk_triples_factory: TriplesFactory,
    combined_dataset: KGDataset,
    alignment_df: pd.DataFrame,
    weight_method: Literal["similarity"],
) -> KGDataset:
    alignment_df["left_ent"] = alignment_df.left.map(lambda x: ds_dataset.entities[x])
    alignment_df["right_ent"] = alignment_df.right.map(
        lambda x: gk_triples_factory.entity_id_to_label[x]
    )

    if weight_method == "similarity":
        alignment_df["w"] = 1 / (1 + alignment_df.dist)
    else:
        raise ValueError()

    weights_dict = {(h, t): w for h, t, w in alignment_df[["left_ent", "right_ent", "w"]].values}

    weights = []
    for h, r, t in combined_dataset.training.triples:
        if r == ExtraRelationGraphPairCombinator.ALIGNMENT_RELATION_NAME:
            w = weights_dict[(h.replace("left:", ""), t.replace("right:", ""))]
            weights.append(w)
        else:
            weights.append(1.0)

    weighted_training = WeightedTriplesFactory(
        combined_dataset.training, weights=torch.Tensor(weights).unsqueeze(1)
    )

    weighted_dataset = KGDataset(
        dataset=EagerDataset(
            weighted_training,
            combined_dataset.testing,
            combined_dataset.validation,
            metadata=combined_dataset.dataset.metadata,
        ),
        evaluation_restricted_entities=combined_dataset.evaluation_restricted_entities,
    )
    return weighted_dataset
