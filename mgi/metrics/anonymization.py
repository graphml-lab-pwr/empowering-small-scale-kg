import numpy as np
import numpy.typing as npt

from mgi.data.datasets.kgdataset import KGDataset


def acc_at_k(
    ds_dataset: KGDataset, gk_dataset: KGDataset, gk_neighbours_ids: npt.NDArray[np.float32], k: int
) -> float:
    assert k <= gk_neighbours_ids.shape[1]

    ds_entities = ds_dataset.entities
    gk_neighbours = np.vectorize(lambda x: gk_dataset.id_to_entity[x])(gk_neighbours_ids)

    assert len(ds_entities) == len(gk_neighbours_ids)

    counter = 0
    for e_ds, e_gk_neighbours in zip(ds_entities, gk_neighbours):
        positions = np.where(e_gk_neighbours == e_ds)[0]
        assert len(positions) <= 1
        if positions.size > 0 and positions[0] < k:
            counter += 1
    return counter / len(ds_entities)
