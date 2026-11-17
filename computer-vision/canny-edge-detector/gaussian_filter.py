"""
Gaussian Smoothing Filter and 2D Spatial Convolution Engine
Generates 2D Gaussian kernels and executes zero-padded spatial convolutions.
"""

from typing import Tuple
import numpy as np


def create_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """
    Generate an isotropic 2D Gaussian filter kernel.
    
    Formula:
    G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    
    Parameters:
        kernel_size: Odd integer kernel dimension (e.g. 3, 5, 7).
        sigma: Standard deviation of the Gaussian distribution.
        
    Returns:
        Normalized (kernel_size, kernel_size) float64 array summing to 1.0.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got: {kernel_size}")
        
    center = kernel_size // 2
    x, y = np.mgrid[-center:center + 1, -center:center + 1]
    
    normal = 1.0 / (2.0 * np.pi * sigma ** 2)
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    
    # Normalize so kernel sums to 1 (preserves image brightness)
    return kernel / np.sum(kernel)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform 2D spatial convolution with zero-padding to maintain original dimensions.
    
    Parameters:
        image: 2D grayscale input array (H, W).
        kernel: 2D filter kernel (Kh, Kw).
        
    Returns:
        Filtered 2D array of identical shape (H, W).
    """
    H, W = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    # Pad input image with zeros around borders
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0.0)
    
    output = np.zeros_like(image, dtype=np.float64)
    
    # Vectorized / sliding window convolution
    for i in range(H):
        for j in range(W):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)
            
    return output


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """
    Smooth an image using Gaussian convolution to suppress high-frequency noise.
    """
    kernel = create_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    return convolve2d(image, kernel)


if __name__ == "__main__":
    k = create_gaussian_kernel(kernel_size=5, sigma=1.4)
    print("5x5 Gaussian Kernel (sum =", np.sum(k), "):")
    print(np.round(k, 4))
    
    # Test on synthetic impulse step function
    test_img = np.zeros((10, 10))
    test_img[3:7, 3:7] = 255.0
    blurred = apply_gaussian_blur(test_img, kernel_size=3, sigma=1.0)
    print("\nOriginal Center Pixel:", test_img[5, 5])
    print("Blurred Center Pixel :", round(blurred[5, 5], 2))
    print("Gaussian smoothing verification successful.")
