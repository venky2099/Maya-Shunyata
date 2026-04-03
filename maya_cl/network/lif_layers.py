# lif_layers.py — SpikingJelly LIF wrappers for Maya-CL

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer
from maya_cl.utils.config import TAU_MEMBRANE, V_THRESHOLD, V_RESET


class ConvLIFBlock(nn.Module):
    """
    Conv2d → BatchNorm → LIF neuron.
    One block = one convolutional spiking layer.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = layer.Conv2d(in_channels, out_channels,
                                 kernel_size=kernel_size, padding=padding, bias=False)
        self.bn   = layer.BatchNorm2d(out_channels)
        self.lif  = neuron.LIFNode(
            tau=TAU_MEMBRANE,
            v_threshold=V_THRESHOLD,
            v_reset=V_RESET,
            detach_reset=True,       # prevents gradient through reset — biologically plausible
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.bn(self.conv(x)))


class FCLIFBlock(nn.Module):
    """
    Linear → LIF neuron.
    One block = one fully-connected spiking layer.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc  = layer.Linear(in_features, out_features, bias=False)
        self.lif = neuron.LIFNode(
            tau=TAU_MEMBRANE,
            v_threshold=V_THRESHOLD,
            v_reset=V_RESET,
            detach_reset=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.fc(x))