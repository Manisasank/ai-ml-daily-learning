# Deep Learning: Multilayer Perceptron From Scratch (NumPy)

An end-to-end, modular Feedforward Deep Neural Network built entirely from scratch in NumPy without relying on deep learning frameworks (PyTorch/TensorFlow).

---

## 📌 Project Overview

This project demystifies the inner workings of deep neural networks by implementing the entire computational graph and gradient mechanics directly in NumPy:
- **Exact Analytical Gradients:** Manual computation of layer-by-layer backpropagation using chain rule matrix calculus.
- **Modern Weight Initialization:** He (Kaiming) normal initialization tailored for ReLU activations and Xavier (Glorot) uniform initialization for Sigmoid/Tanh layers.
- **Momentum Optimizer:** Mini-batch Stochastic Gradient Descent augmented with exponential velocity momentum to escape shallow local minima and dampen oscillations.
- **Non-Linear Decision Boundaries:** Proven convergence on non-linearly separable datasets (Two Moons benchmark) achieving **96.33% Test Accuracy** and **0.9968 ROC-AUC**.

---

## 🏗️ Architecture & Computational Graph

```
deep-learning/neural-network-from-scratch/
├── activations_and_losses.py   # ReLU, Sigmoid, Softmax, BinaryCrossEntropyLoss
├── layers.py                   # Dense layer with He/Xavier weights & momentum update
├── network.py                  # NeuralNetwork container, fit(), forward(), backward()
├── train_and_evaluate.py       # Benchmark training, evaluation metrics, & coordinate inference
└── README.md                   # Mathematical derivations and documentation
```

### Layer Specification
$$\text{Input } (D=2) \xrightarrow[\text{ReLU}]{\text{Dense}(32)} H_1(32) \xrightarrow[\text{ReLU}]{\text{Dense}(16)} H_2(16) \xrightarrow[\text{Sigmoid}]{\text{Dense}(1)} \hat{y} \in [0, 1]$$

---

## 📐 Mathematical Derivations of Backpropagation

### 1. Forward Propagation (Layer $l$)
$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[l]} = \sigma(Z^{[l]})$$

### 2. Backward Error Propagation
Given loss $L = \text{BCE}(\hat{y}, y)$, we backpropagate sensitivity vectors:
$$dZ^{[l]} = dA^{[l]} \odot \sigma'(Z^{[l]})$$
$$\frac{\partial L}{\partial W^{[l]}} = (A^{[l-1]})^T \cdot dZ^{[l]}$$
$$\frac{\partial L}{\partial b^{[l]}} = \sum_{i=1}^{N} dZ_{i,:}^{[l]}$$
$$\frac{\partial L}{\partial A^{[l-1]}} = dZ^{[l]} \cdot (W^{[l]})^T$$

### 3. Momentum Gradient Update
$$v_t = \beta \cdot v_{t-1} - \eta \cdot \nabla_W L$$
$$W_{t} = W_{t-1} + v_t$$

---

## 📊 Benchmark Results (Two Moons Dataset)

| Metric | Result |
| :--- | :---: |
| **Test Accuracy** | **96.33%** |
| **ROC-AUC Score** | **0.9968** |
| **Precision** | **0.9542** |
| **Recall** | **0.9733** |
| **F1-Score** | **0.9637** |

### Confusion Matrix ($N=300$ held-out test samples):
$$\begin{bmatrix} 143 & 7 \\ 4 & 146 \end{bmatrix}$$

---

## 🚀 Quickstart & Usage

### 1. Test Layer Activations and Gradient Derivatives
```bash
python activations_and_losses.py
```

### 2. Verify Dense Layer Transformations
```bash
python layers.py
```

### 3. Run Mini-Batch Gradient Descent
```bash
python network.py
```

### 4. Train on Non-Linear Decision Benchmark & Run Live Coordinate Inference
```bash
python train_and_evaluate.py
```

---

## 🔮 Future Enhancements
- [ ] Implement Adam (Adaptive Moment Estimation) optimizer with bias correction.
- [ ] Add Dropout and $L_2$ weight decay regularization layers.
- [ ] Extend architecture to 1D/2D Convolutional kernels for image filtering.
