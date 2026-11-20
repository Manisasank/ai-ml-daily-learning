"""
Principal Component Analysis (PCA) from Scratch
Implements linear dimensionality reduction via empirical covariance matrix
eigen-decomposition, explained variance ratios, and orthogonal projection.
"""

from typing import Tuple
import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA) dimensionality reduction.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None          # Principal axes eigenvectors (n_components, n_features)
        self.explained_variance_ = None  # Eigenvalues of selected components
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Compute covariance matrix and extract top principal eigenvectors.
        """
        n_samples, n_features = X.shape
        if self.n_components > n_features:
            raise ValueError(f"n_components ({self.n_components}) cannot exceed n_features ({n_features})")
            
        # 1. Mean centering
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. Covariance matrix computation
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # 3. Eigen-decomposition (eigh for symmetric covariance matrices)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. Sort eigenvalues and eigenvectors in descending order
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # 5. Store principal components and variance metrics
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance if total_variance > 0 else np.zeros_like(self.explained_variance_)
        )
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data matrix into lower-dimensional principal component subspace.
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA model is not fitted yet.")
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and project data into principal subspace.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Reconstruct original data space from reduced principal coordinates.
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA model is not fitted yet.")
        return np.dot(X_transformed, self.components_) + self.mean_


if __name__ == "__main__":
    from cluster_generator import generate_cluster_blobs
    
    # 6-dimensional dataset reduced to 2D
    X, _, _ = generate_cluster_blobs(n_samples=800, n_features=6, centers=3, random_state=42)
    
    pca = PCA(n_components=2)
    X_projected = pca.fit_transform(X)
    
    print(f"Original dimensionality : {X.shape[1]}")
    print(f"Reduced dimensionality  : {X_projected.shape[1]}")
    print(f"Explained Variance Ratio: {np.round(pca.explained_variance_ratio_, 4)}")
    print(f"Total Retained Variance : {np.sum(pca.explained_variance_ratio_):.2%}")
    
    # Test inverse reconstruction error
    X_reconstructed = pca.inverse_transform(X_projected)
    reconstruction_mse = np.mean((X - X_reconstructed) ** 2)
    print(f"Reconstruction MSE      : {reconstruction_mse:.4f}")
    print("PCA verification successful.")
