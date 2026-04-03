# seed.py — deterministic run setup

import torch
import numpy as np
import random
from maya_cl.utils.config import SEED


def set_seed(seed: int = SEED) -> None:
    # lock all randomness sources for reproducibility across Papers 1-3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False