"""
Unit tests for Classical Preprocessing Pipeline (PCA & MinMaxScaler).
"""
import numpy as np
import pytest
from src.data.preprocessor import QuantumPreprocessor


def test_pca_dimension_and_bounds():
    # Generate synthetic 100 samples with 50 features
    np.random.seed(42)
    X_train = np.random.randn(100, 50).astype(np.float32)
    X_test = np.random.randn(30, 50).astype(np.float32)

    preprocessor = QuantumPreprocessor(n_components=8, feature_range=(-np.pi, np.pi))
    angles_train = preprocessor.fit_transform(X_train)
    angles_test = preprocessor.transform(X_test)

    # 1. Check dimensions
    assert angles_train.shape == (100, 8), f"Expected shape (100, 8), got {angles_train.shape}"
    assert angles_test.shape == (30, 8), f"Expected shape (30, 8), got {angles_test.shape}"

    # 2. Check bounds [-pi, pi]
    assert np.all(angles_train >= -np.pi - 1e-6), "Train angles violate lower bound -pi"
    assert np.all(angles_train <= np.pi + 1e-6), "Train angles violate upper bound pi"
    assert np.all(angles_test >= -np.pi - 1e-6), "Test angles violate lower bound -pi"
    assert np.all(angles_test <= np.pi + 1e-6), "Test angles violate upper bound pi"

    # 3. Check train min and max are close to -pi and pi
    for col in range(8):
        assert np.isclose(np.min(angles_train[:, col]), -np.pi, atol=1e-5)
        assert np.isclose(np.max(angles_train[:, col]), np.pi, atol=1e-5)

    # 4. Check summary
    summary = preprocessor.get_summary()
    assert summary["n_components"] == 8
    assert summary["total_explained_variance"] > 0.0


def test_no_data_leakage():
    # Verifies that transform does not refit on test set
    np.random.seed(101)
    X_train = np.random.normal(loc=0.0, scale=1.0, size=(100, 20))
    # Test set with completely shifted distribution
    X_test = np.random.normal(loc=10.0, scale=1.0, size=(50, 20))

    preprocessor = QuantumPreprocessor(n_components=8)
    preprocessor.fit(X_train)
    train_mean = preprocessor.pca.mean_

    # Transform test set
    angles_test = preprocessor.transform(X_test)

    # Ensure PCA mean did not change
    assert np.allclose(preprocessor.pca.mean_, train_mean)
    # Ensure clipping kept test angles within [-pi, pi]
    assert np.all(angles_test <= np.pi + 1e-6)
