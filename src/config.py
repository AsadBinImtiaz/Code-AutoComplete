# Configuration settings for the application
import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
EVAL_DIR = BASE_DIR / "evaluation"
ADAPTER_DIR = MODELS_DIR / "trained_adapter"
BEST_ADAPTER_DIR = ADAPTER_DIR / "best_adapter"

BASELINE_FILE = EVAL_DIR / "baseline_m0_results.json"
TRAINED_FILE = EVAL_DIR / "trained_model_results.json"
OUTPUT_FILE = EVAL_DIR / "model_comparison_report.json"

# Dataset paths
STAGE_A_DIR = DATASETS_DIR / "stage_a"
STAGE_B_DIR = DATASETS_DIR / "stage_b"
UNIFIED_DIR = DATASETS_DIR / "unified"

# Model settings
BASE_MODEL_DIR = MODELS_DIR / "base"
MODEL_NAME = str(BASE_MODEL_DIR / "Qwen/Qwen2.5-Coder-3B")
MAX_LENGTH = 1024
SEED = 42

# Training settings
NUM_ITERATIONS = 10
SAMPLES_PER_ITERATION = 5000
BATCH_SIZE = 1
EPOCHS_PER_ITERATION = 1
VAL_MAX_SAMPLES = 2500

# Training hyperparameters
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 100
SAVE_STEPS = 100

# LpRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj']

# FIM tokens
FIM_PREFIX = '<|fim_prefix|>'
FIM_SUFFIX = '<|fim_suffix|>'
FIM_MIDDLE = '<|fim_middle|>'