"""
End-to-End Deep Neural Network Training & Evaluation Suite
Trains the NumPy MLP on non-linear decision boundary benchmarks and evaluates metrics.
"""

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    from .network import NeuralNetwork
    from .layers import Dense
    from .activations_and_losses import ReLU, Sigmoid, BinaryCrossEntropyLoss
except ImportError:
    from network import NeuralNetwork
    from layers import Dense
    from activations_and_losses import ReLU, Sigmoid, BinaryCrossEntropyLoss


def generate_benchmark_data(n_samples: int = 1000, noise: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two interleaving half circles (Moons dataset) for non-linear classification.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    y = y.reshape(-1, 1).astype(float)
    return X, y


def build_model(input_dim: int = 2) -> NeuralNetwork:
    """
    Construct a 3-layer Multilayer Perceptron:
    Input (2) -> Dense (32, ReLU) -> Dense (16, ReLU) -> Dense (1, Sigmoid)
    """
    nn = NeuralNetwork(loss_fn=BinaryCrossEntropyLoss())
    nn.add(Dense(in_features=input_dim, out_features=32, init_method="he", random_state=42))
    nn.add(ReLU())
    nn.add(Dense(in_features=32, out_features=16, init_method="he", random_state=43))
    nn.add(ReLU())
    nn.add(Dense(in_features=16, out_features=1, init_method="xavier", random_state=44))
    nn.add(Sigmoid())
    return nn


def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Calculate full evaluation metrics on test partition.
    """
    y_proba = model.predict_proba(X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def predict_point(model: NeuralNetwork, x1: float, x2: float) -> Dict[str, Any]:
    """
    Inference helper for arbitrary 2D coordinate input.
    """
    point = np.array([[x1, x2]])
    prob = float(model.predict_proba(point)[0, 0])
    label = 1 if prob >= 0.5 else 0
    return {
        "coordinates": (x1, x2),
        "predicted_class": label,
        "class_1_probability": round(prob, 4),
        "confidence": "High" if prob > 0.8 or prob < 0.2 else "Moderate"
    }


if __name__ == "__main__":
    print("Generating non-linear synthetic benchmark dataset (make_moons)...")
    X, y = generate_benchmark_data(n_samples=1200, noise=0.20, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Dataset split: {X_train.shape[0]} train, {X_test.shape[0]} test samples.")
    
    model = build_model(input_dim=2)
    print("\nTraining Multilayer Perceptron from scratch...")
    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=32,
        lr=0.08,
        momentum=0.9,
        verbose=True
    )
    
    print("\n--- Test Set Evaluation ---")
    metrics = evaluate_model(model, X_test, y_test)
    print(classification_report(y_test, metrics["y_pred"], target_names=["Class 0", "Class 1"]))
    print(f"Test Accuracy: {metrics['accuracy']:.2%}")
    print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
    print(f"F1 Score     : {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
    
    # Test point predictions
    test_points = [(0.0, 0.5), (1.5, -0.2), (-0.8, 0.6)]
    print("\n--- Live Inference on Coordinate Points ---")
    for pt in test_points:
        res = predict_point(model, pt[0], pt[1])
        print(f"Point {res['coordinates']} -> Predicted: Class {res['predicted_class']} (Prob: {res['class_1_probability']:.2%}, Confidence: {res['confidence']})")
