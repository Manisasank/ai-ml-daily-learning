"""
Recursive Decision Tree Classifier from Scratch
Supports arbitrary depth control, minimum sample splits, multi-class predictions,
and Mean Decrease in Impurity (MDI) feature importance tracking.
"""

from typing import Optional, List, Dict, Any
import numpy as np

try:
    from .criterion import gini_impurity, entropy, find_best_split
except ImportError:
    from criterion import gini_impurity, entropy, find_best_split


class Node:
    """
    Representation of a single decision tree node or terminal leaf.
    """
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[int] = None,
        probabilities: Optional[np.ndarray] = None,
        n_samples: int = 0,
        impurity: float = 0.0
    ):
        self.feature = feature          # Index of split feature
        self.threshold = threshold      # Split threshold value
        self.left = left                # Left subtree child node
        self.right = right              # Right subtree child node
        self.value = value              # Majority class label (leaf)
        self.probabilities = probabilities # Class probability distribution
        self.n_samples = n_samples
        self.impurity = impurity

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """
    Binary Classification Tree with pre-pruning heuristics.
    """
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 1e-7,
        criterion: str = "gini",
        max_features: Optional[int] = None,
        random_state: int = 42
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        
        self.root: Optional[Node] = None
        self.n_classes_: int = 0
        self.classes_: np.ndarray = np.array([])
        self.feature_importances_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Build the decision tree recursively from training data.
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
        
        rng = np.random.RandomState(self.random_state)
        self.root = self._build_tree(X, y, depth=0, rng=rng)
        
        # Normalize feature importances so they sum to 1.0
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
            
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.RandomState) -> Node:
        """
        Recursive construction of tree nodes.
        """
        n_samples, n_features = X.shape
        unique_labels, counts = np.unique(y, return_counts=True)
        majority_class = unique_labels[np.argmax(counts)]
        
        # Compute class probabilities
        probs = np.zeros(self.n_classes_)
        for lbl, cnt in zip(unique_labels, counts):
            idx = np.where(self.classes_ == lbl)[0][0]
            probs[idx] = cnt / n_samples
            
        current_impurity = gini_impurity(y) if self.criterion == "gini" else entropy(y)
        
        # Base stopping conditions: pure node, depth limit reached, or min samples threshold
        if (len(unique_labels) == 1 or
            depth >= self.max_depth or
            n_samples < self.min_samples_split):
            return Node(
                value=majority_class,
                probabilities=probs,
                n_samples=n_samples,
                impurity=current_impurity
            )
            
        # Select subset of features (useful for Random Forest)
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = rng.choice(n_features, size=self.max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)
            
        best_feat, best_thresh, best_gain = find_best_split(
            X, y, feature_indices=feature_indices, criterion=self.criterion
        )
        
        # Check if valid split exists and meets impurity threshold
        if best_feat is None or best_gain < self.min_impurity_decrease:
            return Node(
                value=majority_class,
                probabilities=probs,
                n_samples=n_samples,
                impurity=current_impurity
            )
            
        # Accumulate feature importance weighted by sample proportion
        self.feature_importances_[best_feat] += best_gain * (n_samples)
        
        # Partition data into left and right branches
        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1, rng)
        
        return Node(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            n_samples=n_samples,
            impurity=current_impurity
        )

    def _traverse_tree(self, x: np.ndarray, node: Node) -> Node:
        """
        Traverse down tree to find appropriate terminal leaf for single sample.
        """
        if node.is_leaf:
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict discrete class labels for array of samples.
        """
        if self.root is None:
            raise ValueError("Model is not fitted yet.")
        return np.array([self._traverse_tree(x, self.root).value for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for array of samples.
        """
        if self.root is None:
            raise ValueError("Model is not fitted yet.")
        return np.array([self._traverse_tree(x, self.root).probabilities for x in X])


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(iris.data, iris.target)
    preds = tree.predict(iris.data)
    acc = np.mean(preds == iris.target)
    
    print(f"Iris Training Accuracy: {acc:.2%}")
    assert acc >= 0.98, "Tree should achieve >= 98% on Iris dataset."
    print("Feature Importances:", np.round(tree.feature_importances_, 4))
    print("Decision Tree verification successful.")
