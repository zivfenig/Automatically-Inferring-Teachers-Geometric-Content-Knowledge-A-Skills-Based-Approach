# ============================================================================
# Configuration for Baseline MLT (Van Hiele Classification - No Skills)
# ============================================================================

import os
from pathlib import Path

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
BASE_MODEL = "google/gemma-3-4b-it"
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
HEAD_DROPOUT = 0.25

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
VALIDATION_SPLIT = 0.15
MAX_SEQUENCE_LENGTH = 2048
SEED = 42
LR_SCHEDULER_TYPE = "cosine"
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
USE_BF16 = True

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
FOLD_ID = 5
QUESTION_COL = "question"
ANSWER_COL = "answer"
LABEL_COL = "final_decision"

# ============================================================================
# PATHS
# ============================================================================
DATA_BASE = Path(__file__).parent / ".." / "Data-and-preprocess"
FOLDS_DIR = DATA_BASE / "HE_Van_Hiele_Dataset" / "folds"
FOLD_TRAIN_CSV = FOLDS_DIR / f"fold_{FOLD_ID}_train.csv"
FOLD_TEST_CSV = FOLDS_DIR / f"fold_{FOLD_ID}_test.csv"

RESULTS_BASE = Path(__file__).parent / "results" / "baseline"
FOLD_OUTPUT_DIR = RESULTS_BASE / f"fold_{FOLD_ID}"
CHECKPOINTS_DIR = FOLD_OUTPUT_DIR / "checkpoints"
MODEL_DIR = FOLD_OUTPUT_DIR / "model"
PREDICTIONS_DIR = FOLD_OUTPUT_DIR / "predictions"

# ============================================================================
# DATALOADER CONFIGURATION
# ============================================================================
DATALOADER_NUM_WORKERS = 0
DATALOADER_PIN_MEMORY = False
SAVE_SAFETENSORS = False