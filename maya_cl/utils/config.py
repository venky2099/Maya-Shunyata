# config.py — Maya-Shunyata (Paper 8) hyperparameters
# Carries forward P7 (Maya-Manas) base. Adds Karma + Shunyata pruning.
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

SEED = 42
T_STEPS = 4
CONV1_CHANNELS = 64
CONV2_CHANNELS = 64
CONV3_CHANNELS = 128
FC1_SIZE = 2048
NUM_CLASSES = 100
TAU_MEMBRANE = 2.0
V_THRESHOLD = 0.3
V_RESET = 0.0

TAU_SHRADDHA = 10.0
TAU_BHAYA = 3.0
TAU_VAIRAGYA = 20.0
TAU_SPANDA = 5.0
TAU_VIVEKA = 50.0
TAU_BUDDHI = 200.0

HEBBIAN_LR = 0.01
LABILITY_INIT = 1.0
LABILITY_PAIN_BOOST = 5.0
LABILITY_DECAY_RATE = 0.95
PAIN_CONFIDENCE_THRESHOLD = 0.25

VAIRAGYA_DECAY_RATE = 0.002315
VAIRAGYA_PROTECTION_THRESHOLD = 0.3
VAIRAGYA_ACCUMULATE_RATE = 0.0015
VAIRAGYA_PAIN_EROSION_RATE = 0.005

VIVEKA_CONSISTENCY_RISE = 0.01
VIVEKA_CONSISTENCY_DECAY = 0.005
VIVEKA_GAIN_MAX = 3.0
VIVEKA_MIN_TASKS = 2

USE_ORTHOGONAL_HEAD = False
PROTOTYPE_DIM = 2048
NUM_TASKS = 10
CLASSES_PER_TASK = 10
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20
REPLAY_BUFFER_SIZE = 50
REPLAY_RATIO = 0.3
REPLAY_VAIRAGYA_PARTIAL_LIFT = 0.8
REPLAY_PAIN_EXEMPT = True
CIL_BOUNDARY_DECAY = 0.50
CIL_MAX_VFOUT_PROTECTION = 0.70

# Chitta — carried from P6/P7, unchanged
CHITTA_SAMSKARA_RISE = 0.002315
CHITTA_SAMSKARA_DECAY = 0.0007
CHITTA_MOHA_THRESHOLD = 0.95
CHITTA_MOHA_RELEASE_RATE = 0.60
CHITTA_MIN_TASKS = 1
CHITTA_GATE_STRENGTH = 0.30

# Manas — carried from P7, unchanged
A_MANAS = 0.10
MANAS_GANE_PEAK_THRESHOLD = 0.5
MANAS_MIN_TASKS = 0

# Karma — P8 new contribution
# Absolute integral of per-synapse weight trajectory across tasks.
# High Karma = chronic cross-task interference = C3 complement tag.
KARMA_ACCUMULATE_RATE = 1.0          # direct |w_t - w_prev| accumulation
KARMA_THRESHOLD = 0.05               # canonical pruning threshold
KARMA_THRESHOLD_LOW = 0.03           # ablation E — aggressive, spike starvation risk
KARMA_THRESHOLD_HIGH = 0.75          # ablation — conservative
KARMA_DECAY_RATE = 0.002315          # ORCID magic number — passive decay between tasks
KARMA_MIN_TASKS = 1                  # Karma needs at least one task of history

# Shunyata pruning — zero-masking only, NO sparse tensors
# Pruning fires at task boundaries only (follows P6 constraint)
# Biological ground: microglial phagocytosis via C1q/C3 complement cascade
SHUNYATA_PRUNE_AT_BOUNDARY = True
SHUNYATA_MASK_RECOVERY = False       # P8: permanent pruning. Recovery deferred to future work.

DATA_DIR = "data/"
RESULTS_DIR = "results/"
