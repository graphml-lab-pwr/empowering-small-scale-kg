from typing import Type

from mgi.data.datasets.conceptnet import ConceptNet
from mgi.data.datasets.fb15k237decoded import FB15K237Decoded
from mgi.data.datasets.kgdataset import KGDataset
from mgi.data.datasets.wd50k import WD50K
from mgi.data.datasets.wn18rrdecoded import WN18RRDecoded
from mgi.data.datasets.yago import YAGO310
from mgi.defaults import SAMPLED_DATASETS


def get_gk_dataset(name: str) -> KGDataset:
    if name == "WN18RR":
        return WN18RRDecoded.from_path()
    elif name == "ConceptNet":
        return ConceptNet.from_path()
    elif name == "FB15K237":
        return FB15K237Decoded.from_path()
    elif name == "YAGO310":
        return YAGO310.from_path()
    elif name == "WD50K":
        return WD50K.from_path()
    else:
        raise ValueError()


def get_ds_dataset(ds_dataset_name: str, seed: int) -> KGDataset:
    if ds_dataset_name in {"WN18RR", "FB15K237", "WD50K"}:
        return get_gk_dataset(ds_dataset_name)
    else:
        return get_dataset_cls(ds_dataset_name).from_path(
            SAMPLED_DATASETS / ds_dataset_name / str(seed)
        )


def get_dataset_cls(name: str) -> Type[KGDataset]:
    name = name.split("_")[0]
    if name == "WN18RR":
        return WN18RRDecoded
    elif name == "FB15K237":
        return FB15K237Decoded
    elif name == "WD50K":
        return WD50K
    else:
        raise ValueError()
