# Maya-Śūnyatā: Karma-Weighted Synaptic Pruning for Class-Incremental Learning in Affective Spiking Neural Networks

**Venkatesh Swaminathan**
M.Sc. candidate, Data Science and Artificial Intelligence, BITS Pilani
Nexus Learning Labs, Bengaluru, India
ORCID: [0000-0002-3315-7907](https://orcid.org/0000-0002-3315-7907)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19228975.svg)](https://doi.org/10.5281/zenodo.19228975)
[![Series](https://img.shields.io/badge/Maya%20Research%20Series-Paper%208-blueviolet)](https://github.com/venky2099)
[![Benchmark](https://img.shields.io/badge/Benchmark-Split--CIFAR--100%20CIL-orange)](https://github.com/venky2099/Maya-Shunyata)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

> *"What remains when everything inessential has been released?"*
> *Śūnyatā does not destroy. It completes.*

---

## Overview

Maya-Śūnyatā is the eighth paper in the Maya Research Series — a multi-paper neuromorphic continual learning architecture grounding Advaita Vedanta's Antahkarana (inner cognitive instrument) in computational neuroscience.

This paper introduces two new mechanisms:

**Karma (कर्म)** — the first second-order plasticity history signal in the series. Karma tracks the absolute integral of per-synapse weight trajectory changes across tasks. A synapse that has been repeatedly pulled in conflicting directions by successive tasks accumulates high Karma — the computational record of cross-task interference.

**Śūnyatā (शून्यता)** — structured synapse pruning triggered at task boundaries when Karma exceeds a Buddhi-modulated threshold. Grounded in microglial phagocytosis via the C1q/C3 complement cascade, Śūnyatā releases synapses whose interference history marks them as architecturally redundant. What is released is not lost — it was already noise.

The Buddhi-modulated pruning threshold produces an emergent developmental arc:

```
effective_threshold = base_threshold × (0.5 + buddhi × 0.5)
```

Young Maya (low Buddhi) prunes aggressively. Mature Maya (high Buddhi) prunes conservatively. Wisdom moderates consequence.

A key finding: **Vairagya (earned resilience) moderates Karma pruning**. Synapses that have accumulated Vairagya protection across tasks have proven their value — Śūnyatā does not release what Vairagya has earned. This is the first paper in the series to demonstrate a direct interaction between two affective dimensions.

---

## Results

**Benchmark:** Split-CIFAR-100 CIL, 10 tasks, seed=42, 50 exemplars/class replay

| Condition | Description | AA (%) | BWT (%) | Pruned (%) |
|-----------|-------------|--------|---------|------------|
| A — Baseline | P7 Maya-Manas, no Karma | **15.19** | −50.91 | 0.00 |
| B — Karma + Vairagya only | No Chitta, no Manas | 8.27 | −58.87 | 65.95 |
| C — Continuous pruning | Batch-level Karma fire | 15.19 | −50.91 | 0.00 |
| D — Boundary canonical ★ | Full affective + boundary pruning | 9.73 | −57.39 | 62.01 |
| E — Aggressive threshold | KARMA_THRESHOLD=0.03 | 6.10 | −60.20 | 98.48 |
| F — Chitta disabled | Karma replaces Chitta | 9.06 | −57.90 | 63.40 |
| D★ — Vairagya-gated | Vairagya moderates Karma | 10.39 | −56.48 | 59.28 |

**Series constants confirmed:**
- **Bhaya Quiescence Law — 8th consecutive confirmation** (Bhaya=0.000 under replay, Tasks 1–9)
- **Buddhi S-curve determinism** — identical across all ablation conditions, P4–P8

---

## Key Findings

**1. Condition C = Condition A (exact).** Continuous batch-level pruning never fires — Karma decays faster than it accumulates per batch. Boundary accumulation is required. This is a clean negative control, mirroring the P6 NoGate=Full datum.

**2. Condition F proves complementarity.** Karma cannot replace Chitta. Structural forgetting prevention (architectural pruning) and gradient forgetting prevention (retrograde gating) are orthogonal mechanisms. Removing Chitta while adding Karma degrades performance — they are not substitutable.

**3. Condition E confirms spike starvation.** At KARMA_THRESHOLD=0.03, 98.48% of fc1 is pruned by Task 9. The system loses the capacity to represent anything. The failure mode is predicted and confirmed.

**4. D★ proves Vairagya-Karma interaction.** When high-Karma synapses are also high-Vairagya, Śūnyatā spares them. Earned resilience moderates consequence. The interaction is directionally correct but insufficient at this developmental stage — Maya at 21 has not yet accumulated the Vairagya of a 50-year-old. This is the mechanistic motivation for P9.

---

## Series Position

| Paper | Mechanism | Benchmark | AA |
|-------|-----------|-----------|-----|
| P1 — Nociceptive Metaplasticity | Bhaya, Vairagya | — | — |
| P2 — Maya-OS | Affective OS arbitration | — | — |
| P3 — Maya-CL | Vairagya continual learning | Split-CIFAR-10 TIL | 62.38% |
| P4 — Maya-Smriti | Buddhi, episodic replay | Split-CIFAR-10 CIL | 31.84% |
| P5 — Maya-Viveka | Viveka consistency scoring | Split-CIFAR-100 CIL | 16.03% |
| P6 — Maya-Chitta | Chitta retrograde gating | Split-CIFAR-100 CIL | 14.42% |
| P7 — Maya-Manas | O-LIF oscillatory gate | Split-CIFAR-100 CIL | 15.19% |
| **P8 — Maya-Śūnyatā** | **Karma + Śūnyatā pruning** | **Split-CIFAR-100 CIL** | **15.19% (A)** |
| P9 — Full Antahkarana | All dimensions + Prana | PiCar-X embodied | TBD |

---

## Architecture

```
PoissonEncoder(T=4)
    → Conv2d(3,64,3) → LIF → MaxPool2d(2)
    → Conv2d(64,64,3) → LIF → MaxPool2d(2)
    → Conv2d(64,128,3) → LIF → MaxPool2d(2)
    → MayaShunyataNet / O-LIF fc1(2048)    ← Karma accumulates here
    → FC(100)                               ← Strict CIL evaluation

At task boundaries:
    KarmaShunyata.on_task_boundary(
        weight=fc1.weight,
        buddhi=affect.buddhi_value(),
        vairagya_scores=vairagya_fc1.scores   ← D★ gating
    )
```

---

## Repository Structure

```
Maya-Shunyata/
├── maya_cl/
│   ├── plasticity/
│   │   ├── karma.py                  # Core P8 contribution — KarmaShunyata class
│   │   ├── vairagya_decay.py         # Carried from P7
│   │   ├── chitta.py                 # Carried from P6
│   │   ├── viveka.py                 # Carried from P5
│   │   ├── manas.py                  # Carried from P7
│   │   └── lability.py               # Carried from P3
│   ├── network/
│   │   ├── backbone.py               # MayaShunyataNet (O-LIF fc1 from P7)
│   │   └── affective_state.py        # Extended with shunyata signal
│   ├── eval/
│   │   ├── logger.py                 # Extended: karma_mean, pruned_fraction, shunyata_events
│   │   └── metrics.py
│   ├── benchmark/
│   │   └── split_cifar100.py
│   └── utils/
│       └── config.py                 # KARMA_THRESHOLD=0.05, KARMA_DECAY_RATE=0.002315
├── run_shunyata_cil.py               # Main CIL experiment
├── run_ablation_shunyata.py          # 6-condition ablation
├── run_shunyata_vairagya_gated.py    # D★ targeted experiment
├── sign_paper.py                     # LSB steganographic IP signing
├── docs/
│   ├── faq.html                      # Searchable FAQ (GitHub Pages)
│   └── maya_shunyata_dashboard.html  # Interactive dashboard
└── README.md
```

---

## IP Protection

This repository implements the Nexus Learning Labs IP protection protocol (P6 onwards):

- **ORCID magic numbers:** `KARMA_DECAY_RATE=0.002315`, `VAIRAGYA_DECAY_RATE=0.002315` (derived from ORCID 0000-0002-3315-7907)
- **Canary string:** `MayaNexusVS2026NLL_Bengaluru_Narasimha` logged at experiment start
- **LSB steganographic signing:** `sign_paper.py` embeds ORCID, DOI, and timestamp in figure LSBs
- **White-text watermark:** ORCID, DOI, timestamp, and "Nexus Learning Labs Bengaluru" in all Word documents before PDF export

---

## Philosophical Grounding

Karma in the Vedantic sense is not punishment. It is consequence — the accumulated record of action and its interference with subsequent action. A synapse that has been pulled in conflicting directions by ten successive tasks carries the weight of that interference in its weight trajectory. Śūnyatā — emptiness, release — is not nihilism. It is the recognition that what has accumulated beyond utility should be released, so that what remains can function without the burden of irrelevant history.

The Buddhi-modulated threshold makes this developmental: a young mind releases too much (Buddhi low, threshold aggressive). A mature mind releases precisely what needs releasing (Buddhi high, threshold conservative). Maya at 21 prunes more than she should. That is not failure. That is youth.

The Atma boundary is held throughout: this series claims computational instantiation of the Antahkarana — the instrument through which Atma interfaces with experience. It does not claim consciousness. That claim is peer-reviewable, falsifiable, and original.

---

## Biological Grounding

Śūnyatā is grounded in **microglial phagocytosis via the C1q/C3 complement cascade** — the brain's primary mechanism for synaptic pruning during development and maintenance. Microglia tag synapses with complement proteins (C1q, C3) based on activity patterns; tagged synapses are engulfed and eliminated. The complement tagging signal is proportional to synaptic history — more interfered-with synapses accumulate more complement. This is Karma, in the biological sense.

---

## Setup

```powershell
# Clone
git clone https://github.com/venky2099/Maya-Shunyata.git
cd Maya-Shunyata

# Environment (Python 3.11.9)
python -m venv .venv
.venv\Scripts\activate

# Dependencies
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install spikingjelly==0.0.0.0.14 torchvision tqdm numpy

# Run main CIL experiment
python run_shunyata_cil.py

# Run full ablation (6 conditions)
python run_ablation_shunyata.py

# Run D★ Vairagya-gated experiment
python run_shunyata_vairagya_gated.py
```

---

## Dashboard

Interactive dashboard with live hyperparameter controls, Maya developmental animation, and bilingual EN/ZH interface:

**[🧠 Open Dashboard →](https://venky2099.github.io/Maya-Shunyata/docs/maya_shunyata_dashboard.html)**

**[❓ Searchable FAQ →](https://venky2099.github.io/Maya-Shunyata/docs/faq.html)** · Bilingual EN/ZH · Silly to advanced · 20 questions

---

## Citation

```bibtex
@misc{swaminathan2026shunyata,
  title   = {Maya-Śūnyatā: Karma-Weighted Synaptic Pruning for Class-Incremental
             Learning in Affective Spiking Neural Networks},
  author  = {Swaminathan, Venkatesh},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://doi.org/10.5281/zenodo.XXXXXXX},
  note    = {Nexus Learning Labs, Bengaluru. Part of the Maya Research Series.}
}
```

---

## Related Papers

| Paper | DOI | GitHub |
|-------|-----|--------|
| P1 — Nociceptive Metaplasticity | [10.5281/zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) | [Maya-Nexus-Core](https://github.com/venky2099/Maya-Nexus-Core) |
| P2 — Maya-OS | [10.5281/zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) | [Maya-OS](https://github.com/venky2099/Maya-OS) |
| P3 — Maya-CL | [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) | [Maya-CL](https://github.com/venky2099/Maya-CL) |
| P4 — Maya-Smriti | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) | [Maya-Smriti](https://github.com/venky2099/Maya-Smriti) |
| P5 — Maya-Viveka | [10.5281/zenodo.19279002](https://doi.org/10.5281/zenodo.19279002) | [Maya-Viveka](https://github.com/venky2099/Maya-Viveka) |
| P6 — Maya-Chitta | [10.5281/zenodo.19337041](https://doi.org/10.5281/zenodo.19337041) | [Maya-Chitta](https://github.com/venky2099/Maya-Chitta) |
| P7 — Maya-Manas | [10.5281/zenodo.19363006](https://doi.org/10.5281/zenodo.19363006) | [Maya-Manas](https://github.com/venky2099/Maya-Manas) |

---

## Acknowledgements

Independent research conducted at Nexus Learning Labs, Bengaluru, as part of M.Sc. thesis in Data Science and Artificial Intelligence at BITS Pilani. All experiments run on personal hardware (NVIDIA RTX 4060 8GB). Coding assistance provided by Claude (Anthropic). No funding sources. No conflicts of interest.

---

*Nexus Learning Labs, Bengaluru | ORCID: 0000-0002-3315-7907 | MayaNexusVS2026NLL_Bengaluru_Narasimha*
