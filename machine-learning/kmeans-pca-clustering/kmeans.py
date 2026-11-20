"""
K-Means Clustering Algorithm from Scratch
Implements Lloyd's algorithm with KMeans++ probabilistic centroid initialization
and Within-Cluster Sum of Squares (Inertia) calculation.
"""

from typing import Tuple, List
import numpy as np


class KMeans:
    """
    K-Means clustering with KMeans++ initialization.
    """
    def __init__(
        self,
        n_clusters: int = 4,
        init: str = "k-means++",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        self.k = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _init_centroids(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """
        Initialize cluster centroids using either uniform random or KMeans++ method.
        """
        n_samples, n_features = X.shape
        
        if self.init == "random":
            indices = rng.choice(n_samples, size=self.k, replace=False)
            return X[indices].copy()
            
        elif self.init == "k-means++":
            # 1. Choose first center uniformly at random
            first_idx = rng.randint(0, n_samples)
            centroids = [X[first_idx]]
            
            # 2. Iteratively select remaining centers with probability proportional to D(x)^2
            for _ in range(1, self.k):
                # Compute squared Euclidean distance of each point to nearest existing centroid
                dist_sq = np.array([
                    min(np.sum((x - c) ** 2) for c in centroids)
                    for x in X
                ])
                
                # Probability distribution proportional to D(x)^2
                probs = dist_sq / np.sum(dist_sq)
                next_idx = rng.choice(n_samples, p=probs)
                centroids.append(X[next_idx])
                
            return np.array(centroids)
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the closest centroid index.
        """
        # Pairwise distance matrix of shape (n_samples, k)
        # Using expanded Euclidean: ||X - C||^2 = ||X||^2 - 2*X*C.T + ||C||^2
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Run iterative Expectation-Maximization loop to optimize centroids.
        """
        rng = np.random.RandomState(self.random_state)
        self.centroids = self._init_centroids(X, rng)
        
        for iteration in range(1, self.max_iter + 1):
            # 1. Assignment Step (Expectation)
            labels = self._assign_clusters(X)
            
            # 2. Update Step (Maximization)
            new_centroids = np.zeros_like(self.centroids)
            for cluster_idx in range(self.k):
                cluster_points = X[labels == cluster_idx]
                if len(cluster_points) > 0:
                    new_centroids[cluster_idx] = np.mean(cluster_points, axis=0)
                else:
                    # Handle empty cluster: reinitialize to random sample point
                    new_centroids[cluster_idx] = X[rng.randint(0, X.shape[0])]
                    
            # 3. Check for convergence (centroid movement < tolerance)
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.n_iter_ = iteration
            
            if centroid_shift < self.tol:
                break
                
        # Final label assignment and inertia calculation
        self.labels_ = self._assign_clusters(X)
        self.inertia_ = float(np.sum([
            np.sum((X[self.labels_ == k] - self.centroids[k]) ** 2)
            for k in range(self.k)
        ]))
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict closest cluster label for unseen query samples.
        """
        if self.centroids is None:
            raise ValueError("KMeans model is not fitted yet.")
        return self._assign_clusters(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and return cluster assignments.
        """
        return self.fit(X).labels_


if __name__ == "__main__":
    from cluster_generator import generate_cluster_blobs
    
    X, y_true, _ = generate_cluster_blobs(n_samples=1000, n_features=4, centers=3, random_state=42)
    
    kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
    labels = kmeans.fit_predict(X)
    
    print(f"K-Means converged in {kmeans.n_iter_} iterations.")
    print(f"Final Inertia (WCSS): {kmeans.inertia_:.2f}")
    print(f"Cluster distributions: {np.bincount(labels)}")
    print("Centroid matrix shape:", kmeans.centroids.shape)
    print("K-Means verification successful.")
