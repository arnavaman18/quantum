"""
Classical Preprocessing Pipeline: Train-Only PCA and Angle MinMaxScaler.
Guarantees strict prevention of data leakage.
"""
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from src.config import N_COMPONENTS_PCA, ANGLE_RANGE


class QuantumPreprocessor:
    """
    Classical preprocessing pipeline:
    1. Fit PCA strictly on training data, reducing to N_COMPONENTS_PCA (8).
    2. Fit MinMaxScaler strictly on training PCA features, scaling to [-pi, pi].
    3. Transform both train and test without data leakage.
    """

    def __init__(self, n_components: int = N_COMPONENTS_PCA, feature_range: Tuple[float, float] = ANGLE_RANGE):
        self.n_components = n_components
        self.feature_range = feature_range
        self.pca = PCA(n_components=n_components, random_state=42)
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.is_fitted = False
        self.explained_variance_ratio_: np.ndarray = None
        self.total_explained_variance_: float = 0.0

    def fit(self, X_train: np.ndarray) -> "QuantumPreprocessor":
        """
        Fit PCA and MinMaxScaler strictly on training data.
        """
        # 1. Fit PCA
        pca_features = self.pca.fit_transform(X_train)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.total_explained_variance_ = float(np.sum(self.explained_variance_ratio_))

        # 2. Fit MinMaxScaler on PCA features
        self.scaler.fit(pca_features)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA and MinMaxScaler.
        Clips output to feature_range to ensure valid rotation angles.
        """
        if not self.is_fitted:
            raise RuntimeError("QuantumPreprocessor must be fitted on training data before transforming.")

        # 1. Project onto top principal components
        pca_features = self.pca.transform(X)

        # 2. Scale to [-pi, pi]
        scaled_angles = self.scaler.transform(pca_features)

        # 3. Safe clipping to prevent out-of-bounds angles from unseen outliers
        clipped_angles = np.clip(scaled_angles, self.feature_range[0], self.feature_range[1])
        return clipped_angles.astype(np.float64)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Fit on X_train and return transformed X_train.
        """
        return self.fit(X_train).transform(X_train)

    def get_summary(self) -> Dict[str, Any]:
        """
        Return metadata summary of PCA and scaling.
        """
        return {
            "n_components": self.n_components,
            "feature_range": self.feature_range,
            "total_explained_variance": self.total_explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist() if self.explained_variance_ratio_ is not None else [],
        }
