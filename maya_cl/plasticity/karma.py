# karma.py — Karma-Shunyata mechanism (Paper 8 core contribution)
# V2: Vairagya-gated pruning — high Vairagya synapses protected from Karma pruning
# Biological grounding: microglial phagocytosis via C1q/C3 complement cascade
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import torch
from maya_cl.utils.config import (
    KARMA_THRESHOLD,
    KARMA_DECAY_RATE,
    KARMA_MIN_TASKS,
    SHUNYATA_PRUNE_AT_BOUNDARY,
    VAIRAGYA_PROTECTION_THRESHOLD,
)


class KarmaShunyata:
    def __init__(self, shape: tuple, device: torch.device,
                 threshold: float = KARMA_THRESHOLD):
        self.shape     = shape
        self.device    = device
        self.threshold = threshold
        self.scores    = torch.zeros(shape, device=device)
        self.mask      = torch.ones(shape, device=device)
        self._tasks_seen   = 0
        self._total_pruned = 0

    def accumulate(self, w_current: torch.Tensor,
                   w_prev: torch.Tensor) -> None:
        with torch.no_grad():
            delta = torch.abs(w_current - w_prev)
            self.scores += delta * self.mask

    def on_task_boundary(self, weight: torch.Tensor,
                         buddhi: float = 1.0,
                         vairagya_scores: torch.Tensor = None) -> int:
        """
        Vairagya-gated Karma pruning.

        A synapse is pruned only if:
          (a) Karma score exceeds Buddhi-modulated threshold
          (b) Vairagya score is below protection threshold — not yet proven resilient

        This is the philosophically correct interaction:
        earned detachment (Vairagya) moderates the consequence of interference (Karma).
        A synapse that has survived enough tasks to earn Vairagya protection
        is not released by Shunyata — it has demonstrated it can carry the weight.
        """
        if not SHUNYATA_PRUNE_AT_BOUNDARY:
            return 0
        if self._tasks_seen < KARMA_MIN_TASKS:
            self._tasks_seen += 1
            return 0

        with torch.no_grad():
            effective_threshold = self.threshold * (0.5 + buddhi * 0.5)

            high_karma = (self.scores > effective_threshold) & (self.mask == 1.0)

            if vairagya_scores is not None:
                # Only prune synapses that are BOTH high-Karma AND low-Vairagya
                # High-Vairagya synapses have earned protection — Karma cannot touch them
                low_vairagya = vairagya_scores < VAIRAGYA_PROTECTION_THRESHOLD
                prune_mask   = high_karma & low_vairagya
            else:
                prune_mask = high_karma

            n_pruned = int(prune_mask.sum().item())

            if n_pruned > 0:
                self.mask[prune_mask]  = 0.0
                weight[prune_mask]     = 0.0
                self._total_pruned    += n_pruned
                frac = prune_mask.float().mean().item() * 100
                vp   = (vairagya_scores >= VAIRAGYA_PROTECTION_THRESHOLD).float().mean().item() * 100 if vairagya_scores is not None else 0.0
                print(f"  [Shunyata★] Task {self._tasks_seen}: "
                      f"{n_pruned} synapses pruned ({frac:.2f}%) | "
                      f"Buddhi={buddhi:.3f} | threshold={effective_threshold:.4f} | "
                      f"Vairagya-protected={vp:.1f}% (spared)")
            else:
                print(f"  [Shunyata★] Task {self._tasks_seen}: 0 pruned — "
                      f"all high-Karma synapses Vairagya-protected")

            self.scores *= (1.0 - KARMA_DECAY_RATE)
            self.scores.clamp_(0.0, None)
            self._tasks_seen += 1

        return n_pruned

    def apply_mask(self, weight: torch.Tensor) -> None:
        with torch.no_grad():
            weight.mul_(self.mask)

    def karma_mean(self) -> float:
        return float(self.scores.mean().item())

    def karma_max(self) -> float:
        return float(self.scores.max().item())

    def pruned_fraction(self) -> float:
        return float((self.mask == 0.0).float().mean().item())

    def active_fraction(self) -> float:
        return float(self.mask.mean().item())

    def total_pruned(self) -> int:
        return self._total_pruned

    def summary(self) -> dict:
        return {
            "karma_mean":      round(self.karma_mean(), 6),
            "karma_max":       round(self.karma_max(), 6),
            "pruned_fraction": round(self.pruned_fraction(), 6),
            "active_fraction": round(self.active_fraction(), 6),
            "total_pruned":    self._total_pruned,
            "tasks_seen":      self._tasks_seen,
        }
