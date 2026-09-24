"""
Central configuration for Quantum Kernel Benchmarking on MedMNIST.
"""
from pathlib import Path
import numpy as np

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data and Artifact directories
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
GRAM_DIR = RESULTS_DIR / "gram_matrices"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

for path in [DATA_DIR, RESULTS_DIR, GRAM_DIR, LOGS_DIR, FIGURES_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset specifications
DATASETS_CONFIG = {
    "pneumoniamnist": {
        "name": "PneumoniaMNIST",
        "task": "binary-class",
        "n_classes": 2,
        "is_rgb": False,
        "train_samples": 500,
        "test_samples": 200,
        "description": "Pediatric chest X-rays (normal vs pneumonia)",
    },
    "breastmnist": {
        "name": "BreastMNIST",
        "task": "binary-class",
        "n_classes": 2,
        "is_rgb": False,
        "train_samples": 546,  # full training set
        "test_samples": 156,   # full test set
        "description": "Breast ultrasound images (malignant vs normal/benign, imbalanced)",
    },
    "retinamnist": {
        "name": "RetinaMNIST",
        "task": "multi-class",
        "n_classes": 5,
        "is_rgb": True,
        "train_samples": 500,
        "test_samples": 200,
        "description": "Retina fundus photographs (5-level diabetic retinopathy grading)",
    },
    "bloodmnist": {
        "name": "BloodMNIST",
        "task": "multi-class",
        "n_classes": 8,
        "is_rgb": True,
        "train_samples": 500,
        "test_samples": 200,
        "description": "Microscopic blood cell images (8 individual cell types)",
    },
}

# Quantum Feature Map parameters
N_QUBITS = 8
N_COMPONENTS_PCA = 8
ANGLE_RANGE = (-np.pi, np.pi)

# Patch configurations for Local Subsystem Kernels
PATCH_CONFIGS = {
    "1-qubit": [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
    "2-qubit": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "2-qubit-cyclic": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)],
    "4-qubit": [(0, 1, 2, 3), (4, 5, 6, 7)],
}

# Default multiscale convex combination weights
DEFAULT_ALPHA = 0.5
ALPHA_SWEEP_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# Sample size scaling values
SAMPLE_SIZE_VALUES = [100, 250, 500]

# Random seeds for multi-seed statistical significance
DEFAULT_SEEDS = [42, 101, 2024]
