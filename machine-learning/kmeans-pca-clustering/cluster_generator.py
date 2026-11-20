"""
Multidimensional Synthetic Cluster Generator
Synthesizes complex Gaussian mixture distributions in high-dimensional space
for clustering and dimensionality reduction benchmarks.
"""

from typing import Tuple
import numpy as np


def generate_cluster_blobs(
    n_samples: int = 1500,
    n_features: int = 6,
    centers: int = 4,
    cluster_std: float = 1.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic clusters with distinct centroid means and covariance spreads.
    
    Parameters:
        n_samples: Total number of data points.
        n_features: Dimensionality of feature space.
        centers: Number of true underlying clusters.
        cluster_std: Standard deviation dispersion of each cluster.
        random_state: Reproducibility seed.
        
    Returns:
        X: Data matrix of shape (n_samples, n_features).
        y: True cluster labels of shape (n_samples,).
        true_centroids: Centroid coordinates of shape (centers, n_features).
    """
    rng = np.random.RandomState(random_state)
    
    # Generate random well-separated centroid positions
    true_centroids = rng.uniform(-10.0, 10.0, size=(centers, n_features))
    
    samples_per_center = [n_samples // centers] * centers
    for i in range(n_samples % centers):
        samples_per_center[i] += 1
        
    X_list = []
    y_list = []
    
    for idx, (centroid, n) in enumerate(zip(true_centroids, samples_per_center)):
        points = rng.normal(loc=centroid, scale=cluster_std, size=(n, n_features))
        X_list.append(points)
        y_list.append(np.full(n, idx, dtype=int))
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle points
    indices = rng.permutation(n_samples)
    return X[indices], y[indices], true_centroids


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Center features to mean 0 and scale to unit variance (StandardScaler).
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  # Prevent division by zero
    
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


if __name__ == "__main__":
    X, y, centroids = generate_cluster_blobs(n_samples=1200, n_features=6, centers=4)
    print("Generated synthetic dataset:")
    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print(f"Number of clusters: {len(np.unique(y))}")
    print(f"Feature means: {np.round(np.mean(X, axis=0), 2)}")
    
    X_scaled, mean, std = standardize_features(X)
    print(f"Scaled feature means: {np.round(np.mean(X_scaled, axis=0), 2)}")
    print(f"Scaled feature stds : {np.round(np.std(X_scaled, axis=0), 2)}")
    print("Cluster generator verification successful.")
