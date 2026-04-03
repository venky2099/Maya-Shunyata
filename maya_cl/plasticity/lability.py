# lability.py — nociceptive metaplasticity
# L[i][j] matrix: per-synapse lability multiplier
# Pain event → offensive lability spike (elevates, not dampens)
# This is the inversion of TACOS defensive metaplasticity

import torch
from maya_cl.utils.config import (
    LABILITY_INIT,
    LABILITY_PAIN_BOOST,
    LABILITY_DECAY_RATE,
)


class LabilityMatrix:
    """
    Maintains a per-synapse lability multiplier for one weight layer.
    Baseline = 1.0 (normal Hebbian rate).
    Pain event: selected synapses spike to LABILITY_PAIN_BOOST.
    Each batch: all values decay exponentially back toward 1.0.
    """

    def __init__(self, shape: tuple, device: torch.device):
        # shape = (out_features, in_features) matching layer.weight
        self.matrix = torch.ones(shape, device=device) * LABILITY_INIT
        self.device = device

    def inject_pain(self, active_mask: torch.Tensor) -> None:
        """
        Elevates lability on synapses connected to currently active pathways.
        active_mask: boolean tensor, same shape as self.matrix.
        Called when TaskSequencer.check_pain_signal() returns True.

        This is offensive metaplasticity:
        pain → hyper-plasticity → rapid pathway rewriting.
        Contrast with TACOS: pain → dampening → consolidation rigidity.
        """
        with torch.no_grad():
            self.matrix[active_mask] = LABILITY_PAIN_BOOST

    def decay(self) -> None:
        """
        Per-batch exponential decay back toward baseline 1.0.
        Ensures lability spike is transient — not permanent destabilisation.
        """
        with torch.no_grad():
            self.matrix = (
                self.matrix * LABILITY_DECAY_RATE
                + LABILITY_INIT * (1.0 - LABILITY_DECAY_RATE)
            )

    def get(self) -> torch.Tensor:
        return self.matrix