# hebbian.py — pure Hebbian weight update, no autograd
# Runs inside torch.no_grad() — never touches the computational graph
# Pattern from HLOP (Xiao et al., ICLR 2024)

import torch
import torch.nn as nn
from maya_cl.utils.config import HEBBIAN_LR


def hebbian_update(layer: nn.Module,
                   pre_spikes: torch.Tensor,
                   post_spikes: torch.Tensor,
                   lability_matrix: torch.Tensor) -> None:
    """
    Applies Hebbian weight update directly to layer.weight.data.
    ΔW = η × lability × (pre ⊗ post)

    Args:
        layer:            nn.Linear or nn.Conv2d — weight to update
        pre_spikes:       [T, B, in_features] — spikes entering the layer
        post_spikes:      [T, B, out_features] — spikes leaving the layer
        lability_matrix:  [out_features, in_features] — per-synapse lability
    """
    with torch.no_grad():
        # average over T timesteps and B batch dimension
        pre_mean  = pre_spikes.mean(dim=(0, 1))    # [in_features]
        post_mean = post_spikes.mean(dim=(0, 1))   # [out_features]

        # outer product — correlation between pre and post activity
        delta_w = torch.outer(post_mean, pre_mean) # [out, in]

        # lability gates the effective learning rate per synapse
        effective_lr = HEBBIAN_LR * lability_matrix

        # apply directly to weights — bypasses autograd entirely
        layer.weight.data += effective_lr * delta_w

        # soft weight clamp — prevents runaway excitation (Chen et al., 2013)
        layer.weight.data.clamp_(-1.0, 1.0)