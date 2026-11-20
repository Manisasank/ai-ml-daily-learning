# Unsupervised Learning: K-Means Clustering & Principal Component Analysis (PCA) From Scratch

An end-to-end mathematical implementation of **K-Means Clustering** (with KMeans++ initialization) and **Principal Component Analysis (PCA)** built entirely in NumPy without Scikit-Learn dependencies.

---

## 📌 Project Overview

Unsupervised learning discovers latent structures, clusters, and low-dimensional manifolds in unlabeled multi-dimensional data. This project implements:
1. **Multivariate Gaussian Mixtures Generator:** Synthesizes well-separated anisotropic clusters across 6-dimensional feature spaces.
2. **K-Means Clustering from Scratch:** Expectation-Maximization loop with $k$-Means++ probabilistic seed initialization, Euclidean distance matrices, and inertia tracking.
3. **Principal Component Analysis (PCA) from Scratch:** Mean-centered empirical sample covariance matrix computation, Hermitian eigen-decomposition, explained variance ratios, and orthogonal projection.
4. **Validation & Diagnostic Suite:** Automated Elbow Method inflection finder, Silhouette coefficient scoring, and 2D ASCII PCA cluster projections.

---

## 🏗️ Architecture

```
machine-learning/kmeans-pca-clustering/
├── cluster_generator.py      # Multivariate Gaussian cluster generator & standard scaler
├── kmeans.py                 # K-Means algorithm with KMeans++ initialization & inertia
├── pca.py                    # PCA with covariance matrix eigen-decomposition & reconstruction
├── evaluate_clustering.py    # Elbow method analysis, silhouette scoring, & ASCII projection
└── README.md                 # Mathematical derivations, formulation, and usage guide
```

---

## 📐 Mathematical Formulations

### 1. K-Means Objective Function (Inertia / WCSS)
Minimizes within-cluster sum of squared Euclidean deviations from cluster centroids:
$$J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \|x_i - \mu_j\|_2^2$$

#### Centroid Maximization Step:
$$\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i$$

### 2. $k$-Means++ Probabilistic Initialization
To prevent poor local minima, initial centers are chosen sequentially with probability proportional to the squared distance from the closest existing center:
$$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$

### 3. Principal Component Analysis (PCA)
Given zero-centered data matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, compute sample covariance matrix:
$$\mathbf{\Sigma} = \frac{1}{N - 1} \mathbf{X}^T \mathbf{X}$$

Find eigenvalues $\lambda_i$ and orthonormal eigenvectors $\mathbf{v}_i$ satisfying:
$$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

Project data to $k$-dimensional orthogonal subspace ($k \le D$):
$$\mathbf{Z} = \mathbf{X} \mathbf{V}_k$$

Explained variance ratio for component $i$:
$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{D} \lambda_j}$$

### 4. Silhouette Metric
For sample $i$ with mean intra-cluster distance $a(i)$ and mean nearest-cluster distance $b(i)$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, +1]$$

---

## 📊 Benchmark Results (6D Synthetic Dataset)

| Metric | $k=2$ | $k=3$ | $k=4$ (Optimal) | $k=5$ | $k=6$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Inertia (WCSS)** | 2,926.05 | 1,755.09 | **394.93** (Elbow) | 367.55 | 333.59 |
| **Silhouette Score** | 0.5501 | 0.5531 | **0.7292** (Peak) | 0.6144 | 0.4988 |

### PCA Dimensionality Reduction:
- **Principal Component 1:** Captures **68.22%** of total variance.
- **Principal Component 2:** Captures **20.77%** of total variance.
- **2D Subspace Cumulative Variance:** **88.99%** of original 6D information preserved.

---

## 🚀 Quickstart & Usage

### 1. Test Cluster Generator & Standardization
```bash
python cluster_generator.py
```

### 2. Run K-Means Clustering Algorithm
```bash
python kmeans.py
```

### 3. Run PCA Dimensionality Reduction & Reconstruction
```bash
python pca.py
```

### 4. Run Elbow Analysis, Silhouette Scoring & 2D Projection
```bash
python evaluate_clustering.py
```

---

## 🔮 Future Enhancements
- [ ] Implement DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
- [ ] Add t-SNE (t-Distributed Stochastic Neighbor Embedding) non-linear manifold projection.
- [ ] Implement Gaussian Mixture Models (GMM) with soft Expectation-Maximization probabilities.
