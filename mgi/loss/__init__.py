from typing import Type

from pykeen.losses import Loss, NSSALoss

from mgi.loss.weighted import WeightedNSSALoss


def get_loss(loss: str) -> Type[Loss]:
    if loss == "nssa":
        return NSSALoss
    elif loss == "weighted_nssa":
        return WeightedNSSALoss
    else:
        raise ValueError(f"Unknown loss: {loss}.")
