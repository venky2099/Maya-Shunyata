# chitta.py -- Chitta-Samskara implicit memory mechanism (Paper 6, carried forward P7)
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import torch
from maya_cl.utils.config import (
    CHITTA_GATE_STRENGTH,
    CHITTA_SAMSKARA_RISE,
    CHITTA_SAMSKARA_DECAY,
    CHITTA_MOHA_THRESHOLD,
    CHITTA_MOHA_RELEASE_RATE,
    CHITTA_MIN_TASKS,
)


class ChittaSamskara:
    def __init__(self, shape: tuple, device: torch.device):
        self.shape  = shape
        self.device = device
        self.traces = torch.zeros(shape, device=device)
        self._tasks_seen = 0

    def update(self, active_mask: torch.Tensor) -> None:
        with torch.no_grad():
            self.traces[active_mask]  += CHITTA_SAMSKARA_RISE
            self.traces[~active_mask] -= CHITTA_SAMSKARA_DECAY
            self.traces.clamp_(0.0, 1.0)

    def compute_gradient_gate(self, active_mask: torch.Tensor,
                              tasks_seen: int) -> torch.Tensor:
        with torch.no_grad():
            if self._tasks_seen < CHITTA_MIN_TASKS:
                return torch.ones(self.shape, device=self.device)
            active_float = active_mask.float()
            gate = 1.0 - (self.traces * active_float * CHITTA_GATE_STRENGTH)
            return gate.clamp(0.0, 1.0)

    def apply_gradient_gate(self, grad: torch.Tensor,
                            gate: torch.Tensor) -> None:
        with torch.no_grad():
            grad.mul_(gate)

    def detect_moha(self) -> torch.Tensor:
        with torch.no_grad():
            return self.traces >= CHITTA_MOHA_THRESHOLD

    def apply_moha_release(self, moha_mask: torch.Tensor) -> None:
        with torch.no_grad():
            if not moha_mask.any():
                return
            self.traces[moha_mask] *= CHITTA_MOHA_RELEASE_RATE
            n = int(moha_mask.sum().item())
            print(f"  [Moha release: {n} synapses ({moha_mask.float().mean()*100:.2f}%) loosened]")

    def on_task_boundary(self) -> None:
        with torch.no_grad():
            self.traces *= 0.90
            self._tasks_seen += 1

    def chitta_activity(self, active_mask: torch.Tensor) -> float:
        with torch.no_grad():
            if not active_mask.any():
                return 0.0
            return float(self.traces[active_mask].mean().item())

    def mean_samskara(self) -> float:
        return float(self.traces.mean().item())

    def moha_fraction(self) -> float:
        return float((self.traces >= CHITTA_MOHA_THRESHOLD).float().mean().item())

    def high_samskara_fraction(self, threshold: float = 0.5) -> float:
        return float((self.traces >= threshold).float().mean().item())
