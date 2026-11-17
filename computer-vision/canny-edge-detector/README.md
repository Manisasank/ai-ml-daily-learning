# Canny Edge Detector & Spatial Convolution Engine (From Scratch)

A complete, mathematical implementation of the classical **Canny Edge Detection Algorithm** in pure Python and NumPy without OpenCV or third-party computer vision libraries.

---

## 📌 Overview

Edge detection is a foundational task in Computer Vision, forming the structural basis for feature extraction, contour analysis, object segmentation, and classical lane/boundary detection. Developed by John F. Canny in 1986, the Canny algorithm optimizes three critical criteria:
1. **Low Error Rate:** Mark genuine edges without hallucinating spurious noise.
2. **Precise Localization:** Detected edges must lie as close as possible to true physical boundaries.
3. **Single Response:** A single boundary must produce exactly one 1-pixel wide line response.

---

## 🏗️ Architecture & Pipeline Stages

```
computer-vision/canny-edge-detector/
├── gaussian_filter.py     # 2D Gaussian kernel generation and zero-padded spatial convolution
├── sobel_gradients.py     # Sobel derivative kernels, gradient magnitude and orientation angles
├── non_max_suppression.py # Gradient-direction interpolation and 1-pixel edge thinning
├── hysteresis.py          # Double thresholding and 8-connected component hysteresis tracking
├── pipeline.py            # End-to-end Canny detector with synthetic generator and ASCII visualizer
└── README.md              # Mathematical derivations and documentation
```

---

## 📐 Mathematical Formulation

### 1. Gaussian Smoothing (Noise Reduction)
An isotropic 2D Gaussian filter kernel is convolved across the image:
$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

### 2. Sobel Directional Derivatives
Convolving with horizontal ($K_x$) and vertical ($K_y$) Sobel kernels:
$$K_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad K_y = \begin{bmatrix} +1 & +2 & +1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix}$$

$$G = \sqrt{I_x^2 + I_y^2}$$
$$\theta = \arctan2(I_y, I_x) \times \frac{180^\circ}{\pi}$$

### 3. Non-Maximum Suppression (NMS)
Thins edge ridges along the gradient normal $\theta$ by quantizing into 4 principal directions:
- **$0^\circ$ (Horizontal gradient):** Compare against West and East neighbors.
- **$45^\circ$ (Diagonal NE-SW):** Compare against North-East and South-West neighbors.
- **$90^\circ$ (Vertical gradient):** Compare against North and South neighbors.
- **$135^\circ$ (Diagonal NW-SE):** Compare against North-West and South-East neighbors.

### 4. Double Thresholding & Hysteresis
Pixels are partitioned using two dynamic cutoffs ($T_{\text{low}}$ and $T_{\text{high}}$):
- **Strong Edges:** $G(x, y) \ge T_{\text{high}}$
- **Weak Edges:** $T_{\text{low}} \le G(x, y) < T_{\text{high}}$
- **Suppressed (Noise):** $G(x, y) < T_{\text{low}}$

**Hysteresis Rule:** A weak edge pixel is promoted to strong if and only if at least one of its 8-connected neighbors is a strong edge pixel.

---

## 🚀 Quickstart & Usage

### 1. Test Gaussian Kernel & 2D Convolution
```bash
python gaussian_filter.py
```

### 2. Test Sobel Directional Derivatives
```bash
python sobel_gradients.py
```

### 3. Test Non-Maximum Suppression (Edge Thinning)
```bash
python non_max_suppression.py
```

### 4. Test Double Thresholding & Hysteresis
```bash
python hysteresis.py
```

### 5. Run Full End-to-End Pipeline & ASCII Visualizer
```bash
python pipeline.py
```

---

## 🔮 Future Enhancements
- [ ] Add Bilateral Filtering to preserve edges while smoothing flat color regions.
- [ ] Implement Hough Transform for geometric line and circle detection on top of Canny edges.
- [ ] Implement Harris Corner Detection operator.
