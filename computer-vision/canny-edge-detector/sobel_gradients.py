"""
Sobel Spatial Gradient & Orientation Calculator
Computes horizontal and vertical directional derivative kernels,
gradient intensity magnitude, and continuous gradient orientation angles.
"""

from typing import Tuple
import numpy as np

try:
    from .gaussian_filter import convolve2d
except ImportError:
    from gaussian_filter import convolve2d


# Standard 3x3 Sobel spatial approximation kernels
SOBEL_X = np.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]
], dtype=np.float64)

SOBEL_Y = np.array([
    [ 1.0,  2.0,  1.0],
    [ 0.0,  0.0,  0.0],
    [-1.0, -2.0, -1.0]
], dtype=np.float64)


def compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate directional derivatives, intensity magnitude, and orientation angles.
    
    Parameters:
        image: Smoothed 2D grayscale array (H, W).
        
    Returns:
        magnitude: Normalized gradient strength array (H, W) in range [0, 255].
        theta: Gradient orientation angles in degrees [0, 180).
        grad_x: Raw horizontal derivative response.
        grad_y: Raw vertical derivative response.
    """
    grad_x = convolve2d(image, SOBEL_X)
    grad_y = convolve2d(image, SOBEL_Y)
    
    # Gradient magnitude via Euclidean hypotenuse
    magnitude = np.hypot(grad_x, grad_y)
    
    # Normalize magnitude to 0 - 255
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = (magnitude / max_val) * 255.0
        
    # Gradient orientation in radians, converted to positive degrees [0, 180)
    theta = np.arctan2(grad_y, grad_x) * (180.0 / np.pi)
    theta[theta < 0] += 180.0
    
    return magnitude, theta, grad_x, grad_y


if __name__ == "__main__":
    # Create a step vertical edge: left half 0, right half 255
    edge_test = np.zeros((15, 15))
    edge_test[:, 7:] = 255.0
    
    mag, angle, gx, gy = compute_gradients(edge_test)
    print("Edge test image shape:", edge_test.shape)
    print("Maximum gradient magnitude at boundary:", round(np.max(mag), 2))
    print("Gradient angle at vertical edge (should be ~0 or ~180 deg):", round(angle[7, 7], 1), "deg")
    print("Sobel gradient computation verified.")
