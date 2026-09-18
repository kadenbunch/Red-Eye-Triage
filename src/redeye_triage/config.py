"""Global configuration for the red-eye triage pipeline.

All hyperparameters below are the values used for the results reported in the
manuscript (CLAIM checklist item 33: reproducibility). Paths can be overridden
with environment variables so the same code runs locally or in Google Colab:

    REDEYE_DATA_DIR    folder containing one sub-folder per class (default: data/images)
    REDEYE_OUTPUT_DIR  folder where all tables, figures and checkpoints are written
                       (default: outputs)
    REDEYE_FAST_MODE   set to "1" for a quick end-to-end debug run (NOT for results)
"""

import os
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
DATASET_PATH = os.environ.get("REDEYE_DATA_DIR", str(REPO_ROOT / "data" / "images"))
OUTPUT_DIR = os.environ.get("REDEYE_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PER_MODEL_DIR = os.path.join(OUTPUT_DIR, "per_model_results")
EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explainability")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparison_figures")
LLM_DIR = os.path.join(OUTPUT_DIR, "llm_outputs")
KNOWLEDGE_BASE_PATH = os.environ.get(
    "REDEYE_KB_PATH", str(REPO_ROOT / "knowledge_base" / "knowledge_base.json"))

# ------------------------------------------------------------------ #
# Task definition
# ------------------------------------------------------------------ #
# Order matters: label index = position in this list.
CLASSES = ["Inflammatory", "Eyelid", "Normal", "Hemorrhage"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ------------------------------------------------------------------ #
# Experiment settings
# ------------------------------------------------------------------ #
SEED = 42
FAST_MODE = os.environ.get("REDEYE_FAST_MODE", "0") == "1"
MODELS_TO_RUN = ["efficientnet", "mobilenet", "resnet18"]

if FAST_MODE:  # smoke-test settings only
    N_FOLDS = 2
    EPOCHS = 1
    N_BOOTSTRAP = 100
    MAX_IMAGES_PER_CLASS = 30
    TEST_SIZE = 0.30
    BATCH_SIZE = 16
    LR = 1e-4
else:  # settings used for the manuscript
    N_FOLDS = 5
    EPOCHS = 15
    N_BOOTSTRAP = 2000
    MAX_IMAGES_PER_CLASS = None
    TEST_SIZE = 0.15
    BATCH_SIZE = 32
    LR = 1e-4

# LLM settings
GEMINI_MODEL = os.environ.get("REDEYE_GEMINI_MODEL", "gemini-3.5-flash")
LLM_TEMPERATURE = 0.2


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    """Deterministic seeding across Python, NumPy, PyTorch and CUDA (CLAIM item 33)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_output_dirs():
    for d in (OUTPUT_DIR, CKPT_DIR, PER_MODEL_DIR):
        os.makedirs(d, exist_ok=True)
