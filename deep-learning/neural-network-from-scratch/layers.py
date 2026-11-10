"""
Fully Connected (Dense) Layer Implementation with Analytical Backpropagation
Supports He and Xavier weight initialization and momentum updates.
"""

from typing import Tuple
import numpy as np


class Dense:
    """
    Dense (Fully Connected) Layer:
    Forward: Z = X @ W + b
    Backward:
      dW = X.T @ dZ
      db = sum(dZ, axis=0, keepdims=True)
      dX = dZ @ W.T
    """
    def __init__(self, in_features: int, out_features: int, init_method: str = "he", random_state: int = 42):
        rng = np.random.RandomState(random_state)
        
        if init_method == "he":
            # He Normal: std = sqrt(2 / in_features) - ideal for ReLU
            std = np.sqrt(2.0 / in_features)
            self.weights = rng.randn(in_features, out_features) * std
        elif init_method == "xavier":
            # Xavier Uniform: limit = sqrt(6 / (in_features + out_features)) - ideal for Sigmoid/Tanh
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.weights = rng.uniform(-limit, limit, (in_features, out_features))
        else:
            self.weights = rng.randn(in_features, out_features) * 0.01
            
        self.biases = np.zeros((1, out_features))
        
        # Caches for backpropagation
        self.inputs = None
        self.d_weights = None
        self.d_biases = None
        
        # Momentum velocity buffers
        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.biases)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute affine transformation Z = X @ W + b.
        """
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Compute gradients with respect to weights, biases, and input activations.
        """
        self.d_weights = np.dot(self.inputs.T, d_out)
        self.d_biases = np.sum(d_out, axis=0, keepdims=True)
        d_inputs = np.dot(d_out, self.weights.T)
        return d_inputs

    def update_params(self, lr: float, momentum: float = 0.9):
        """
        Apply Stochastic Gradient Descent with Momentum.
        v = momentum * v - lr * grad
        w = w + v
        """
        self.v_w = momentum * self.v_w - lr * self.d_weights
        self.weights += self.v_w
        
        self.v_b = momentum * self.v_b - lr * self.d_biases
        self.biases += self.v_b


if __name__ == "__main__":
    layer = Dense(in_features=4, out_features=3, init_method="he")
    x = np.random.randn(5, 4)
    out = layer.forward(x)
    print("Layer weights shape:", layer.weights.shape)
    print("Forward output shape:", out.shape)
    
    d_out = np.random.randn(5, 3)
    d_in = layer.backward(d_out)
    print("d_weights shape:", layer.d_weights.shape)
    print("d_biases shape:", layer.d_biases.shape)
    print("d_inputs shape:", d_in.shape)
    
    layer.update_params(lr=0.01)
    print("Parameters updated successfully with momentum.")
