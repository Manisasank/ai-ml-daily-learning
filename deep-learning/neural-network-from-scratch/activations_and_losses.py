"""
Activation Functions and Loss Functions with Exact Analytical Derivatives
Implemented from scratch using NumPy.
"""

from typing import Tuple
import numpy as np


class Activation:
    """Base class for all activation layers."""
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    """
    Rectified Linear Unit:
    f(x) = max(0, x)
    f'(x) = 1 if x > 0 else 0
    """
    def __init__(self):
        self.inputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        d_inputs = d_out.copy()
        d_inputs[self.inputs <= 0] = 0.0
        return d_inputs


class Sigmoid(Activation):
    """
    Sigmoid Activation:
    f(x) = 1 / (1 + exp(-x))
    f'(x) = f(x) * (1 - f(x))
    """
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Clip inputs to prevent numerical overflow in exp
        clipped = np.clip(inputs, -500, 500)
        self.output = 1.0 / (1.0 + np.exp(-clipped))
        return self.output

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out * self.output * (1.0 - self.output)


class Softmax(Activation):
    """
    Softmax Activation for multi-class classification:
    f(x)_i = exp(x_i) / sum(exp(x_j))
    """
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        exp_vals = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self.output = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        return self.output

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # Softmax backward is typically combined with Categorical Cross-Entropy
        # for numerical stability. Here we provide the general Jacobian vector product.
        batch_size = d_out.shape[0]
        d_inputs = np.empty_like(d_out)
        for i, (single_output, single_d_out) in enumerate(zip(self.output, d_out)):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            d_inputs[i] = np.dot(jacobian, single_d_out)
        return d_inputs


class BinaryCrossEntropyLoss:
    """
    Binary Cross-Entropy Loss:
    L = -1/N * sum(y * log(p) + (1-y) * log(1-p))
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        batch_size = y_pred.shape[0]
        return ((y_pred - y_true) / (y_pred * (1.0 - y_pred))) / batch_size


if __name__ == "__main__":
    # Self-test activations and gradients
    x = np.array([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]])
    relu = ReLU()
    sig = Sigmoid()
    
    print("Inputs:\n", x)
    print("ReLU forward:\n", relu.forward(x))
    print("ReLU backward with gradient 1.0:\n", relu.backward(np.ones_like(x)))
    print("Sigmoid forward:\n", sig.forward(x))
    print("Sigmoid backward with gradient 1.0:\n", sig.backward(np.ones_like(x)))
    
    # Self-test loss
    bce = BinaryCrossEntropyLoss()
    y_t = np.array([[1.0], [0.0]])
    y_p = np.array([[0.9], [0.1]])
    loss = bce.forward(y_p, y_t)
    grad = bce.backward(y_p, y_t)
    print(f"\nBCE Loss: {loss:.5f}")
    print(f"BCE Gradient:\n{grad}")
