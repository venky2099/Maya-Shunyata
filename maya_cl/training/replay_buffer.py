# replay_buffer.py — Class-wise ring buffer for CIL episodic replay
# Paper 4: Affective-Gated Episodic Replay in Maya-CL
#
# Design choice: Ring buffer (class-wise FIFO), NOT reservoir sampling.
# Rationale: guarantees balanced class representation across all 5 tasks.
# Reservoir sampling risks erasing early-task classes as buffer fills —
# catastrophic for a 5-task sequence where Task 0 classes are oldest.
#
# Biological grounding: hippocampal interleaved replay (CLS theory)
# McClelland et al. (1995); Frankland & Bontempi (2005)
# Empirical precedent: SESLR (Sleep Enhanced Latent Replay), ARROW (2025)
#
# Buffer size M is a primary ablation hyperparameter (Paper 4 Section 4).
# Ablate over M ∈ {20, 50, 100, 200} per class.
# Standard: M=50/class = 500 total for Split-CIFAR-10 (10 classes).

import torch
import random
from collections import defaultdict
from maya_cl.utils.config import NUM_CLASSES


class ReplayBuffer:
    """
    Per-class ring buffer. Stores raw (image, label) CPU tensors.

    max_per_class: M exemplars per class
    Total capacity: M × NUM_CLASSES (e.g. 50 × 10 = 500 total)

    Update protocol: called once per epoch AFTER training on current task.
    Sample protocol: called per batch DURING training to build interleaved batch.
    """

    def __init__(self, max_per_class: int = 50):
        self.max_per_class = max_per_class
        # dict: class_id (int) → list of image tensors (CPU, [C, H, W])
        self.buffer: dict[int, list] = defaultdict(list)

    # ── Buffer maintenance ─────────────────────────────────────────────────

    def update(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Add a batch of current-task samples to the buffer.
        Ring eviction: oldest sample removed when class capacity exceeded.

        Args:
            images: [B, C, H, W] — raw pixel images (NOT spike-encoded)
            labels: [B]           — global 0-9 class labels (CIL protocol)

        Call once per epoch after the training loop completes.
        Storing raw images (not spikes) because Poisson encoding is
        stochastic — re-encoding at replay time maintains spike diversity.
        """
        images_cpu = images.detach().cpu()
        labels_cpu = labels.detach().cpu()

        for img, lbl in zip(images_cpu, labels_cpu):
            cls = int(lbl.item())
            self.buffer[cls].append(img.clone())
            # ring eviction — FIFO within class
            if len(self.buffer[cls]) > self.max_per_class:
                self.buffer[cls].pop(0)

    # ── Sampling ───────────────────────────────────────────────────────────

    def sample(self, n_samples: int,
               device: torch.device) -> tuple[torch.Tensor | None,
                                              torch.Tensor | None]:
        """
        Sample n_samples uniformly from all stored classes.
        Uniform over items (not classes) — naturally weights larger classes.

        Returns:
            (images, labels) on target device, or (None, None) if buffer empty.
            images: [n, C, H, W] — raw pixel images, ready for Poisson encoding
            labels: [n]           — global class labels
        """
        all_items = []
        for cls, imgs in self.buffer.items():
            for img in imgs:
                all_items.append((img, cls))

        if len(all_items) == 0:
            return None, None

        n = min(n_samples, len(all_items))
        sampled = random.sample(all_items, n)
        imgs, lbls = zip(*sampled)

        images_t = torch.stack(imgs).to(device)            # [n, C, H, W]
        labels_t = torch.tensor(lbls, dtype=torch.long, device=device)
        return images_t, labels_t

    # ── Diagnostics ────────────────────────────────────────────────────────

    def size(self) -> int:
        """Total number of stored samples across all classes."""
        return sum(len(v) for v in self.buffer.values())

    def class_counts(self) -> dict:
        """Per-class sample count — use for balance verification."""
        return {cls: len(imgs) for cls, imgs in sorted(self.buffer.items())}

    def is_ready(self) -> bool:
        """True once at least one class has samples — safe to start sampling."""
        return self.size() > 0

    def __repr__(self) -> str:
        return (f"ReplayBuffer(max_per_class={self.max_per_class}, "
                f"total={self.size()}, classes={list(self.buffer.keys())})")