"""
Clustering Evaluation Suite & PCA Projection CLI
Evaluates optimal k selection via the Elbow Method, calculates Silhouette Scores,
and renders a 2D ASCII cluster projection chart.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from .cluster_generator import generate_cluster_blobs, standardize_features
    from .kmeans import KMeans
    from .pca import PCA
except ImportError:
    from cluster_generator import generate_cluster_blobs, standardize_features
    from kmeans import KMeans
    from pca import PCA


def calculate_silhouette_score(X: np.ndarray, labels: np.ndarray, sample_size: int = 500) -> float:
    """
    Calculate mean silhouette score across samples.
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) is mean intra-cluster distance, b(i) is mean nearest-cluster distance.
    """
    n_samples = X.shape[0]
    if sample_size and n_samples > sample_size:
        indices = np.random.RandomState(42).choice(n_samples, size=sample_size, replace=False)
        X = X[indices]
        labels = labels[indices]
        n_samples = sample_size
        
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= n_samples:
        return 0.0
        
    silhouette_coefficients = []
    
    for i in range(n_samples):
        point = X[i]
        label = labels[i]
        
        # Intra-cluster points
        same_cluster_points = X[labels == label]
        if len(same_cluster_points) > 1:
            a_i = np.mean([np.linalg.norm(point - p) for p in same_cluster_points if not np.array_equal(p, point)])
        else:
            a_i = 0.0
            
        # Other clusters
        b_i_candidates = []
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_points = X[labels == other_label]
            if len(other_points) > 0:
                dist = np.mean([np.linalg.norm(point - p) for p in other_points])
                b_i_candidates.append(dist)
                
        b_i = min(b_i_candidates) if b_i_candidates else 0.0
        
        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 0 else 0.0
        silhouette_coefficients.append(s_i)
        
    return float(np.mean(silhouette_coefficients))


def run_elbow_analysis(X: np.ndarray, k_range: range = range(2, 8)) -> List[Dict[str, Any]]:
    """
    Compute Inertia and Silhouette scores across candidate values of k.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=42)
        km.fit(X)
        sil = calculate_silhouette_score(X, km.labels_, sample_size=400)
        results.append({
            "k": k,
            "inertia": round(km.inertia_, 2),
            "silhouette": round(sil, 4)
        })
    return results


def render_ascii_pca_scatter(X_pca: np.ndarray, labels: np.ndarray, width: int = 48, height: int = 16) -> str:
    """
    Render 2D PCA projection scatter plot as terminal ASCII.
    """
    grid = [["." for _ in range(width)] for _ in range(height)]
    
    x_min, x_max = np.min(X_pca[:, 0]), np.max(X_pca[:, 0])
    y_min, y_max = np.min(X_pca[:, 1]), np.max(X_pca[:, 1])
    
    cluster_symbols = ["A", "B", "C", "D", "E", "F"]
    
    for pt, lbl in zip(X_pca, labels):
        col = int((pt[0] - x_min) / (x_max - x_min + 1e-9) * (width - 1))
        row = int((y_max - pt[1]) / (y_max - y_min + 1e-9) * (height - 1))
        symbol = cluster_symbols[lbl % len(cluster_symbols)]
        grid[row][col] = symbol
        
    return "\n".join("".join(r) for r in grid)


if __name__ == "__main__":
    print("Generating 6-dimensional synthetic clustering challenge...")
    X, y_true, _ = generate_cluster_blobs(n_samples=1200, n_features=6, centers=4, cluster_std=1.0, random_state=42)
    X_scaled, _, _ = standardize_features(X)
    
    print("\n--- Running Elbow & Silhouette Metric Search (k=2..6) ---")
    elbow_results = run_elbow_analysis(X_scaled, k_range=range(2, 7))
    print(f"{'Clusters (k)':<14s} {'Inertia (WCSS)':<18s} {'Silhouette Score':<18s}")
    print("-" * 50)
    for r in elbow_results:
        print(f"k = {r['k']:<10d} {r['inertia']:<18.2f} {r['silhouette']:<18.4f}")
        
    best_k_result = max(elbow_results, key=lambda r: r["silhouette"])
    optimal_k = best_k_result["k"]
    print(f"\nOptimal Cluster Count (Max Silhouette): k = {optimal_k} (Score: {best_k_result['silhouette']:.4f})")
    
    # Fit champion model and project via PCA
    champion_km = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42)
    clusters = champion_km.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    print("\n--- 2D PCA Cluster Projection Map (A, B, C, D = Clusters) ---")
    print(render_ascii_pca_scatter(X_2d, clusters))
    print(f"Explained Variance across 2 principal axes: {np.sum(pca.explained_variance_ratio_):.2%}")
