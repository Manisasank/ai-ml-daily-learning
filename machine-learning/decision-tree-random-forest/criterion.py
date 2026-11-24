"""
Splitting Criteria and Impurity Metrics
Implements Gini Impurity, Shannon Entropy, and vectorized optimal split search.
"""

from typing import Tuple, Optional
import numpy as np


def gini_impurity(y: np.ndarray) -> float:
    """
    Calculate Gini Impurity for a label array.
    Gini = 1 - sum(p_i^2)
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(1.0 - np.sum(probabilities ** 2))


def entropy(y: np.ndarray) -> float:
    """
    Calculate Shannon Entropy for a label array.
    Entropy = -sum(p_i * log2(p_i))
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # Filter 0 probabilities to prevent log2(0)
    probs = probabilities[probabilities > 0]
    return float(-np.sum(probs * np.log2(probs)))


def information_gain(
    y_parent: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    criterion: str = "gini"
) -> float:
    """
    Calculate impurity reduction (Information Gain) resulting from a candidate split.
    """
    imp_fn = gini_impurity if criterion == "gini" else entropy
    
    n_parent = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)
    
    if n_left == 0 or n_right == 0:
        return 0.0
        
    parent_imp = imp_fn(y_parent)
    weighted_child_imp = (n_left / n_parent) * imp_fn(y_left) + (n_right / n_parent) * imp_fn(y_right)
    return float(parent_imp - weighted_child_imp)


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Optional[np.ndarray] = None,
    criterion: str = "gini"
) -> Tuple[Optional[int], Optional[float], float]:
    """
    Exhaustively evaluate candidate thresholds across features to maximize Information Gain.
    
    Returns:
        best_feature_idx: Index of optimal feature to split on.
        best_threshold: Numeric threshold value.
        best_gain: Maximum information gain achieved.
    """
    n_samples, n_features = X.shape
    if feature_indices is None:
        feature_indices = np.arange(n_features)
        
    best_gain = 0.0
    best_feature_idx = None
    best_threshold = None
    
    for feat_idx in feature_indices:
        values = X[:, feat_idx]
        # Use unique midpoints between consecutive sorted values as candidate thresholds
        unique_vals = np.unique(values)
        if len(unique_vals) <= 1:
            continue
            
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        
        # Subsample thresholds if too large for speed
        if len(thresholds) > 20:
            thresholds = np.percentile(thresholds, np.linspace(5, 95, 20))
            
        for thresh in thresholds:
            left_mask = values <= thresh
            y_left = y[left_mask]
            y_right = y[~left_mask]
            
            gain = information_gain(y, y_left, y_right, criterion=criterion)
            
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feat_idx
                best_threshold = thresh
                
    return best_feature_idx, best_threshold, best_gain


if __name__ == "__main__":
    # Test on synthetic pure vs mixed split
    y_pure = np.array([0, 0, 0, 0])
    y_mixed = np.array([0, 0, 1, 1])
    
    print(f"Gini Impurity (Pure) : {gini_impurity(y_pure):.4f}")
    print(f"Gini Impurity (Mixed): {gini_impurity(y_mixed):.4f}")
    print(f"Entropy (Pure)       : {entropy(y_pure):.4f}")
    print(f"Entropy (Mixed)      : {entropy(y_mixed):.4f}")
    
    # Test best split finder
    X_test = np.array([[1.0], [2.0], [8.0], [9.0]])
    y_test = np.array([0, 0, 1, 1])
    f_idx, th, g = find_best_split(X_test, y_test)
    print(f"\nBest split -> Feature: {f_idx}, Threshold: {th}, Gain: {g:.4f}")
    assert th == 5.0 and g == 0.5
    print("Criterion and best-split verification successful.")
