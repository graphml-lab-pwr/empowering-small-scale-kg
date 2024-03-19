from typing import Type

from pykeen.training import SLCWATrainingLoop

from mgi.training.weighted import WeightedSLCWATrainingLoop


def get_training_loop(
    training_loop: str,
) -> Type[SLCWATrainingLoop] | Type[WeightedSLCWATrainingLoop]:
    if training_loop == "SLCWA":
        return SLCWATrainingLoop
    elif training_loop == "WeightedSLCWA":
        return WeightedSLCWATrainingLoop
    else:
        raise ValueError(f"Unknown training loop: {training_loop}")
