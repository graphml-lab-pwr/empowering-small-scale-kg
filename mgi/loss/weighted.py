from typing import Optional

import torch
from pykeen.losses import (
    NSSALoss,
    UnsupportedLabelSmoothingError,
    prepare_negative_scores_for_softmax,
)
from torch.nn import functional


class WeightedNSSALoss(NSSALoss):
    def process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        batch_filter: Optional[torch.BoolTensor] = None,
        num_entities: Optional[int] = None,
        weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Function is a copy of the original pykeen function with the only difference that the weights are used.

        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)

        negative_scores = prepare_negative_scores_for_softmax(
            batch_filter=batch_filter,  # type: ignore
            negative_scores=negative_scores,
            # we do not allow full -inf rows, since we compute the softmax over this tensor
            no_inf_rows=True,
        )

        # compute weights (without gradient tracking)
        assert negative_scores.ndimension() == 2
        neg_weights = negative_scores.detach().mul(self.inverse_softmax_temperature).softmax(dim=-1)

        # fill negative scores with some finite value, e.g., 0 (they will get masked out anyway)
        negative_scores = torch.masked_fill(  # type: ignore
            negative_scores, mask=~torch.isfinite(negative_scores), value=0.0
        )

        return self(  # type: ignore
            pos_scores=positive_scores,
            neg_scores=negative_scores,
            neg_weights=neg_weights,
            label_smoothing=label_smoothing,
            num_entities=num_entities,
            weights=weights,
        )

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
        neg_weights: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        num_entities: Optional[int] = None,
        weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Function is a copy of the original pykeen function with the only difference that the weights are used.

        """Calculate the loss for the given scores.

        :param pos_scores: shape: s_p
            a tensor of positive scores
        :param neg_scores: shape: s_n
            a tensor of negative scores
        :param neg_weights: shape: s_n
            the adversarial weights of the negative scores
        :param label_smoothing:
            An optional label smoothing parameter.
        :param num_entities:
            The number of entities (required for label-smoothing).

        :returns:
            a scalar loss value
        """
        neg_loss = self.negative_loss_term_unreduced(
            neg_scores=neg_scores, label_smoothing=label_smoothing, num_entities=num_entities
        )
        # note: this is a reduction along the softmax dim; since the weights are already normalized
        #       to sum to one, we want a sum reduction here, instead of using the self._reduction
        neg_loss = (neg_weights * neg_loss).sum(dim=-1)  # type: ignore
        neg_loss = neg_loss * weights  # type: ignore
        neg_loss = self._reduction_method(neg_loss)  # type: ignore

        pos_loss = self.positive_loss_term(
            pos_scores=pos_scores,
            label_smoothing=label_smoothing,
            num_entities=num_entities,
            weights=weights,
        )
        return self.factor * (pos_loss + neg_loss)  # type: ignore

    def positive_loss_term(
        self,
        pos_scores: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        num_entities: Optional[int] = None,
        weights: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Function is a copy of the original pykeen function with the only difference that the weights are used.
        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)
        return -self._reduction_method(functional.logsigmoid(self.margin + pos_scores) * weights)  # type: ignore
