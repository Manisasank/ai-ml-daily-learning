"""
Modular Neural Network Engine
Chains arbitrary layer compositions, executes full forward/backward passes,
and performs mini-batch gradient descent with momentum.
"""

from typing import List, Dict, Any
import numpy as np

try:
    from .layers import Dense
    from .activations_and_losses import ReLU, Sigmoid, BinaryCrossEntropyLoss
except ImportError:
    from layers import Dense
    from activations_and_losses import ReLU, Sigmoid, BinaryCrossEntropyLoss


class NeuralNetwork:
    """
    Feedforward Neural Network with backpropagation engine.
    """
    def __init__(self, loss_fn=None):
        self.layers: List[Any] = []
        self.loss_fn = loss_fn if loss_fn is not None else BinaryCrossEntropyLoss()
        self.loss_history: List[float] = []

    def add(self, layer: Any) -> "NeuralNetwork":
        """
        Append a Dense or Activation layer to the network architecture.
        """
        self.layers.append(layer)
        return self

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagate inputs forward through all network layers.
        """
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Propagate gradients backward through all network layers in reverse order.
        """
        current_grad = loss_grad
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.05,
        momentum: float = 0.9,
        verbose: bool = True
    ) -> List[float]:
        """
        Train network parameters using mini-batch gradient descent with momentum.
        """
        n_samples = X.shape[0]
        self.loss_history = []

        for epoch in range(1, epochs + 1):
            # Shuffle at start of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # 1. Forward pass
                y_pred = self.forward(X_batch)

                # 2. Compute loss & gradient
                loss = self.loss_fn.forward(y_pred, y_batch)
                epoch_losses.append(loss)
                loss_grad = self.loss_fn.backward(y_pred, y_batch)

                # 3. Backward pass
                self.backward(loss_grad)

                # 4. Update parameters
                for layer in self.layers:
                    if hasattr(layer, "update_params"):
                        layer.update_params(lr=lr, momentum=momentum)

            mean_epoch_loss = float(np.mean(epoch_losses))
            self.loss_history.append(mean_epoch_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {mean_epoch_loss:.5f}")

        return self.loss_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability distribution.
        """
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict discrete class labels.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


if __name__ == "__main__":
    # Test on synthetic binary classification problem
    np.random.seed(42)
    X = np.random.randn(200, 2)
    # Target: points outside unit radius are class 1, inside are class 0 (non-linear)
    y = ((X[:, 0]**2 + X[:, 1]**2) > 1.0).astype(float).reshape(-1, 1)

    nn = NeuralNetwork()
    nn.add(Dense(in_features=2, out_features=16, init_method="he", random_state=42))
    nn.add(ReLU())
    nn.add(Dense(in_features=16, out_features=8, init_method="he", random_state=43))
    nn.add(ReLU())
    nn.add(Dense(in_features=8, out_features=1, init_method="xavier", random_state=44))
    nn.add(Sigmoid())

    print("Architecture built. Beginning test training...")
    history = nn.fit(X, y, epochs=50, batch_size=32, lr=0.1, momentum=0.9, verbose=True)
    preds = nn.predict(X)
    acc = np.mean(preds == y)
    print(f"\nFinal Training Accuracy: {acc:.2%}")
    print(f"Initial Loss: {history[0]:.4f} -> Final Loss: {history[-1]:.4f}")
