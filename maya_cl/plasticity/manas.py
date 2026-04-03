# manas.py — Manas oscillatory gate (Paper 7 core contribution)
#
# Manas (????): the oscillating, doubting, sensory-receiving mind.
# First component of the Antahkarana in Advaita Vedanta.
# Characterised by Sankalpa (resolution/receptivity) and Vikalpa (doubt/suppression).
# Manas does not decide — it oscillates and presents.
#
# MECHANISM — Oscillatory LIF (O-LIF):
#   The membrane threshold descends monotonically across T_STEPS timesteps,
#   tracing the Vikalpa-to-Sankalpa transition within each forward pass.
#
#   V_threshold(t) = V_base + A_manas * cos(p * t / (T_STEPS - 1))
#
#   t=0: V_base + A_manas  — Vikalpa (maximum suppression)
#   t=T: V_base - A_manas  — Sankalpa (full receptivity)
#
#   Only salient signals survive the high threshold at t=0.
#   As the window opens, progressively weaker signals are admitted.
#   The gate closes after the presentation ends — not mid-pass.
#
# MANAS-GANE INTERSECTION:
#   Only synapses that are BOTH:
#     (a) spatially consistent — Viveka-qualified (high consistency score)
#     (b) temporally aligned   — spike occurred during peak receptivity phase
#   receive amplified Vairagya protection.
#
#   This prevents noise spikes at t=3 (wide-open window) from hijacking
#   GANE amplification — only signals that survived early suppression qualify.
#
# Biological grounding:
#   Thalamo-cortical oscillations (Steriade et al., 1993):
#   thalamic bursts precede cortical receptivity — the gate opens, inputs
#   flow through, then closes after the presentation ends.
#   The full oscillatory cycle spans multiple input presentations,
#   not a single forward pass.
#
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import torch
import numpy as np
from maya_cl.utils.config import (
    V_THRESHOLD, A_MANAS, T_STEPS,
    MANAS_GANE_PEAK_THRESHOLD,
)


class ManasGate:
    """
    Oscillatory threshold schedule for O-LIF neurons.

    Computes the per-timestep threshold vector for the fc1 LIF layer,
    tracing the half-cycle Vikalpa?Sankalpa descent across T_STEPS.

    Also tracks which timesteps qualify as peak-aligned for Manas-GANE
    intersection — only spikes during high-threshold phases (early timesteps)
    are considered salient enough to earn amplified Vairagya protection.
    """

    def __init__(self, t_steps: int = T_STEPS,
                 v_base: float = V_THRESHOLD,
                 a_manas: float = A_MANAS):
        self.t_steps = t_steps
        self.v_base  = v_base
        self.a_manas = a_manas

        # Precompute threshold schedule — shape [T_STEPS]
        # cos(p * t / (T-1)): 1.0 at t=0, -1.0 at t=T-1
        t = np.arange(t_steps, dtype=np.float32)
        cos_vals = np.cos(np.pi * t / max(t_steps - 1, 1))
        self.threshold_schedule = v_base + a_manas * cos_vals  # [T_STEPS]

        # Peak-alignment mask: timesteps where cos > MANAS_GANE_PEAK_THRESHOLD
        # These are the early high-threshold steps — signals that spike here
        # are genuinely salient, not just benefiting from an open window
        self.peak_mask = cos_vals > MANAS_GANE_PEAK_THRESHOLD  # [T_STEPS] bool

        print(f"[Manas] Threshold schedule: "
              f"{[f'{v:.3f}' for v in self.threshold_schedule]}")
        print(f"[Manas] Peak-aligned steps: {np.where(self.peak_mask)[0].tolist()}")

    def get_threshold(self, t: int) -> float:
        """Return threshold for timestep t."""
        return float(self.threshold_schedule[t])

    def is_peak_aligned(self, t: int) -> bool:
        """True if timestep t is in the high-threshold (salient) phase."""
        return bool(self.peak_mask[t])

    def threshold_tensor(self, device: torch.device) -> torch.Tensor:
        """Full schedule as a tensor for vectorised use."""
        return torch.tensor(self.threshold_schedule, device=device)


class ManasConsistency:
    """
    Per-synapse Manas-GANE intersection tracker.

    Tracks which synapses fired during peak-aligned timesteps across batches.
    A synapse qualifies for amplified Vairagya protection only if it is:
      (a) Viveka-consistent (handled by VivekaConsistency)
      (b) Manas-peak-aligned (handled here)

    The intersection of (a) and (b) is the Manas-GANE hotspot set.
    """

    def __init__(self, shape: tuple, device: torch.device):
        self.shape  = shape
        self.device = device

        # Running score: how often each synapse fires during peak-aligned steps
        # Range [0, 1] — high score = consistently salient, not just window-riding
        self.peak_scores = torch.zeros(shape, device=device)

        self._rise  = 0.01
        self._decay = 0.005

    def update(self, peak_active_mask: torch.Tensor) -> None:
        """
        Update peak-alignment scores from current batch.

        peak_active_mask: bool tensor [out, in]
            True for synapses that were active during peak-aligned timesteps.
            Computed in the training loop by intersecting fc1 activity
            with ManasGate.peak_mask at each timestep.
        """
        with torch.no_grad():
            self.peak_scores[peak_active_mask]  += self._rise
            self.peak_scores[~peak_active_mask] -= self._decay
            self.peak_scores.clamp_(0.0, 1.0)

    def compute_manas_gane_mask(self,
                                viveka_scores: torch.Tensor,
                                viveka_threshold: float = 0.3) -> torch.Tensor:
        """
        Compute the Manas-GANE intersection mask.

        A synapse qualifies if:
          peak_score > 0.3  (fired during salient phase consistently)
          AND
          viveka_score > viveka_threshold  (spatially consistent across tasks)

        Returns:
            bool tensor [out, in] — True = amplified Vairagya protection
        """
        with torch.no_grad():
            manas_qualified  = self.peak_scores > 0.3
            viveka_qualified = viveka_scores > viveka_threshold
            return manas_qualified & viveka_qualified

    def mean_peak_score(self) -> float:
        return float(self.peak_scores.mean().item())

    def gane_eligible_fraction(self, viveka_scores: torch.Tensor,
                               viveka_threshold: float = 0.3) -> float:
        mask = self.compute_manas_gane_mask(viveka_scores, viveka_threshold)
        return float(mask.float().mean().item())

