from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from mgi.data.datasets.kgdataset import KGDataset


class BaseMapping(ABC):
    @property
    @abstractmethod
    def node_map(self) -> dict[str, list[tuple[str, float]]]:
        pass

    @abstractmethod
    def get_neighbours_map_from_dataset(
        self, dataset: KGDataset
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]:
        pass
