from typing import Optional

import torch
from pykeen.losses import Loss
from pykeen.models import Model
from pykeen.training import SLCWATrainingLoop
from pykeen.typing import InductiveMode

from mgi.data.weighted import WeightedSLCWABatch


class WeightedSLCWATrainingLoop(SLCWATrainingLoop):
    @staticmethod
    def _process_batch_static(  # type: ignore
        model: Model,
        loss: Loss,
        mode: Optional[InductiveMode],
        batch: WeightedSLCWABatch,
        start: Optional[int],
        stop: Optional[int],
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        # The function is a copy of the original pykeen function with the only difference that the
        # weights are used.

        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # split batch
        positive_batch, negative_batch, positive_filter, weights = batch

        # send to device
        positive_batch = positive_batch[start:stop].to(device=model.device)  # type: ignore
        negative_batch = negative_batch[start:stop]  # type: ignore
        weights = weights[start:stop].to(device=model.device)
        if positive_filter is not None:
            positive_filter = positive_filter[start:stop]  # type: ignore
            negative_batch = negative_batch[positive_filter]  # type: ignore
            positive_filter = positive_filter.to(model.device)  # type: ignore
        # Make it negative batch broadcastable (required for num_negs_per_pos > 1).
        negative_score_shape = negative_batch.shape[:-1]
        negative_batch = negative_batch.view(-1, 3)  # type: ignore

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(model.device)  # type: ignore

        # Compute negative and positive scores
        positive_scores = model.score_hrt(positive_batch, mode=mode)
        negative_scores = model.score_hrt(negative_batch, mode=mode).view(*negative_score_shape)

        return (
            loss.process_slcwa_scores(  # type: ignore
                positive_scores=positive_scores,
                negative_scores=negative_scores,  # type: ignore
                label_smoothing=label_smoothing,
                batch_filter=positive_filter,
                num_entities=model._get_entity_len(mode=mode),
                weights=weights,
            )
            + model.collect_regularization_term()
        )
