"""
Random Forest Classifier Ensemble from Scratch
Implements Bootstrap Aggregating (Bagging), feature subsampling (sqrt(D)),
Out-of-Bag (OOB) validation scoring, and ensemble probability voting.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np

try:
    from .decision_tree import DecisionTreeClassifier
except ImportError:
    from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest Ensemble constructed from collections of de-correlated Decision Trees.
    """
    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 8,
        min_samples_split: int = 2,
        max_features: Optional[str] = "sqrt",
        criterion: str = "gini",
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        
        self.trees: List[DecisionTreeClassifier] = []
        self.classes_: np.ndarray = np.array([])
        self.n_classes_: int = 0
        self.feature_importances_: np.ndarray = np.array([])
        self.oob_score_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """
        Train ensemble of bootstrap trees and compute Out-of-Bag (OOB) accuracy.
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Determine number of features per split
        if self.max_features == "sqrt":
            k_features = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k_features = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            k_features = min(n_features, self.max_features)
        else:
            k_features = n_features
            
        rng = np.random.RandomState(self.random_state)
        self.trees = []
        self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
        
        # Out-of-bag prediction accumulators
        oob_predictions = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=int)
        
        for tree_idx in range(self.n_estimators):
            # 1. Bootstrap sample (sampling with replacement)
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            
            X_boot = X[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            
            # 2. Build de-correlated tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=k_features,
                criterion=self.criterion,
                random_state=rng.randint(0, 100000)
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            # Accumulate feature importance
            self.feature_importances_ += tree.feature_importances_
            
            # 3. Track Out-of-Bag predictions
            if len(oob_indices) > 0:
                oob_probs = tree.predict_proba(X[oob_indices])
                oob_predictions[oob_indices] += oob_probs
                oob_counts[oob_indices] += 1
                
        # Average feature importance across ensemble
        self.feature_importances_ /= self.n_estimators
        
        # Compute final OOB score across samples evaluated at least once
        valid_oob = oob_counts > 0
        if np.any(valid_oob):
            oob_class_preds = np.argmax(oob_predictions[valid_oob], axis=1)
            mapped_preds = self.classes_[oob_class_preds]
            self.oob_score_ = float(np.mean(mapped_preds == y[valid_oob]))
        else:
            self.oob_score_ = 0.0
            
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ensemble class probabilities via soft voting.
        """
        if not self.trees:
            raise ValueError("Random Forest is not fitted yet.")
        tree_probs = [tree.predict_proba(X) for tree in self.trees]
        return np.mean(tree_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict discrete class labels via majority ensemble vote.
        """
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]


if __name__ == "__main__":
    from sklearn.datasets import load_wine
    wine = load_wine()
    
    rf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
    rf.fit(wine.data, wine.target)
    preds = rf.predict(wine.data)
    train_acc = np.mean(preds == wine.target)
    
    print(f"Number of trees: {len(rf.trees)}")
    print(f"Wine Training Accuracy: {train_acc:.2%}")
    print(f"Out-of-Bag (OOB) Score: {rf.oob_score_:.2%}")
    assert train_acc >= 0.95
    assert rf.oob_score_ >= 0.90
    print("Top 3 Feature Importances:", np.argsort(rf.feature_importances_)[-3:][::-1])
    print("Random Forest verification successful.")
