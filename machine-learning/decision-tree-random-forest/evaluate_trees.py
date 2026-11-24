"""
Tree Evaluation Suite & Decision Tree ASCII Visualizer
Compares Single Decision Tree vs Random Forest Ensemble performance,
ranks MDI feature importances, and prints visual rule trees.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

try:
    from .decision_tree import DecisionTreeClassifier, Node
    from .random_forest import RandomForestClassifier
except ImportError:
    from decision_tree import DecisionTreeClassifier, Node
    from random_forest import RandomForestClassifier


def print_ascii_tree(node: Node, feature_names: List[str], depth: int = 0, prefix: str = "Root: "):
    """
    Recursively print visual ASCII representation of Decision Tree branching rules.
    """
    indent = "  " * depth
    if node.is_leaf:
        probs = [f"{p:.2f}" for p in node.probabilities]
        print(f"{indent}{prefix}--> [LEAF] Class = {node.value} (Samples: {node.n_samples}, Probs: {probs})")
        return
        
    feat_name = feature_names[node.feature] if feature_names else f"Feature {node.feature}"
    print(f"{indent}{prefix}[SPLIT] If {feat_name} <= {node.threshold:.2f} (Impurity: {node.impurity:.3f}, Samples: {node.n_samples})")
    print_ascii_tree(node.left, feature_names, depth + 1, prefix="True:  ")
    print_ascii_tree(node.right, feature_names, depth + 1, prefix="False: ")


def benchmark_tree_models(X_train, X_test, y_train, y_test, feature_names) -> Dict[str, Any]:
    """
    Compare single tree against Random Forest on test set.
    """
    # 1. Single Decision Tree
    tree = DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    acc_tree = accuracy_score(y_test, y_pred_tree)
    f1_tree = f1_score(y_test, y_pred_tree, average="weighted")
    
    # 2. Random Forest Ensemble
    rf = RandomForestClassifier(n_estimators=40, max_depth=6, max_features="sqrt", random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
    
    return {
        "tree_accuracy": acc_tree,
        "tree_f1": f1_tree,
        "rf_accuracy": acc_rf,
        "rf_f1": f1_rf,
        "rf_oob": rf.oob_score_,
        "tree_model": tree,
        "rf_model": rf,
        "feature_importances": rf.feature_importances_
    }


if __name__ == "__main__":
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Dataset: Wine Recognition (Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(target_names)})")
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    
    results = benchmark_tree_models(X_train, X_test, y_train, y_test, feature_names)
    
    print("\n" + "=" * 58)
    print("      MODEL COMPARISON: DECISION TREE VS RANDOM FOREST")
    print("=" * 58)
    print(f"  Single Decision Tree Test Accuracy : {results['tree_accuracy']:.2%} (F1: {results['tree_f1']:.4f})")
    print(f"  Random Forest Test Accuracy        : {results['rf_accuracy']:.2%} (F1: {results['rf_f1']:.4f})")
    print(f"  Random Forest Out-of-Bag (OOB) Score: {results['rf_oob']:.2%}")
    print("=" * 58)
    
    print("\n--- Top 5 Important Features (MDI Impurity Decrease) ---")
    sorted_idx = np.argsort(results["feature_importances"])[::-1]
    for i in range(min(5, len(feature_names))):
        idx = sorted_idx[i]
        print(f"  {feature_names[idx]:24s} : {results['feature_importances'][idx]:.4f}")
        
    print("\n--- Single Decision Tree Branching Architecture ---")
    print_ascii_tree(results["tree_model"].root, feature_names)
