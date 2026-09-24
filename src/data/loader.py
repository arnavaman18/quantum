"""
MedMNIST Dataset Loader with Stratified Sampling and Caching.
"""
from typing import Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import DATASETS_CONFIG, DATA_DIR


def rgb_to_grayscale(images: np.ndarray) -> np.ndarray:
    """
    Convert RGB images (N, 28, 28, 3) to Grayscale (N, 28, 28) using ITU-R BT.601 weights.
    """
    if images.ndim == 4 and images.shape[-1] == 3:
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        return np.tensordot(images, weights, axes=([3], [0]))
    return images


def load_medmnist_dataset(
    dataset_key: str,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    seed: int = 42,
    convert_rgb_to_gray: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a MedMNIST dataset, preprocess images (flattened float32 [0, 1]),
    and optionally apply stratified sampling.

    Args:
        dataset_key: One of 'pneumoniamnist', 'breastmnist', 'retinamnist', 'bloodmnist'
        n_train: Max training samples (if None or >= len, uses all)
        n_test: Max test samples (if None or >= len, uses all)
        seed: Random seed for stratified sampling
        convert_rgb_to_gray: Convert 3-channel RGB to 1-channel grayscale

    Returns:
        X_train, y_train, X_test, y_test
        X are flattened 2D arrays (N, D), y are 1D label arrays (N,)
    """
    import medmnist
    from medmnist import INFO

    dataset_key = dataset_key.lower()
    if dataset_key not in DATASETS_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Expected one of {list(DATASETS_CONFIG.keys())}")

    config = DATASETS_CONFIG[dataset_key]
    info = INFO[dataset_key]
    data_class = getattr(medmnist, info["python_class"])

    # Load official splits
    train_data = data_class(split="train", download=True, root=str(DATA_DIR))
    test_data = data_class(split="test", download=True, root=str(DATA_DIR))

    X_train_raw = train_data.imgs
    y_train_raw = train_data.labels.ravel()

    X_test_raw = test_data.imgs
    y_test_raw = test_data.labels.ravel()

    # Convert RGB to Grayscale if requested
    if convert_rgb_to_gray and config["is_rgb"]:
        X_train_raw = rgb_to_grayscale(X_train_raw)
        X_test_raw = rgb_to_grayscale(X_test_raw)

    # Flatten and normalize to [0, 1]
    X_train_flat = X_train_raw.reshape(X_train_raw.shape[0], -1).astype(np.float32) / 255.0
    X_test_flat = X_test_raw.reshape(X_test_raw.shape[0], -1).astype(np.float32) / 255.0

    # Stratified subsampling for Train
    target_n_train = n_train if n_train is not None else config["train_samples"]
    if target_n_train is not None and target_n_train < len(X_train_flat):
        X_train, _, y_train, _ = train_test_split(
            X_train_flat,
            y_train_raw,
            train_size=target_n_train,
            stratify=y_train_raw,
            random_state=seed,
        )
    else:
        X_train, y_train = X_train_flat, y_train_raw

    # Stratified subsampling for Test
    target_n_test = n_test if n_test is not None else config["test_samples"]
    if target_n_test is not None and target_n_test < len(X_test_flat):
        X_test, _, y_test, _ = train_test_split(
            X_test_flat,
            y_test_raw,
            train_size=target_n_test,
            stratify=y_test_raw,
            random_state=seed,
        )
    else:
        X_test, y_test = X_test_flat, y_test_raw

    return X_train, y_train, X_test, y_test
