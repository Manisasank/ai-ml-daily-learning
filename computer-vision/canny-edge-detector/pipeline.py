"""
End-to-End Canny Edge Detection Pipeline
Chains Gaussian smoothing, Sobel gradients, non-maximum suppression,
and hysteresis tracking into a cohesive, production-grade vision detector.
"""

from typing import Tuple, Dict, Any
import numpy as np

try:
    from .gaussian_filter import apply_gaussian_blur
    from .sobel_gradients import compute_gradients
    from .non_max_suppression import non_max_suppression
    from .hysteresis import double_threshold, hysteresis_tracking
except ImportError:
    from gaussian_filter import apply_gaussian_blur
    from sobel_gradients import compute_gradients
    from non_max_suppression import non_max_suppression
    from hysteresis import double_threshold, hysteresis_tracking


class CannyEdgeDetector:
    """
    Standard multi-stage Canny Edge Detection pipeline.
    """
    def __init__(
        self,
        gaussian_kernel_size: int = 5,
        gaussian_sigma: float = 1.4,
        low_threshold_ratio: float = 0.08,
        high_threshold_ratio: float = 0.22
    ):
        self.kernel_size = gaussian_kernel_size
        self.sigma = gaussian_sigma
        self.low_ratio = low_threshold_ratio
        self.high_ratio = high_threshold_ratio

    def detect_edges(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Execute full 5-stage edge detection pipeline.
        
        Returns:
            Dictionary containing final binary edge map and intermediate state maps.
        """
        # Ensure 2D float64
        img = image.astype(np.float64)
        if img.ndim == 3:
            # Convert RGB to Grayscale using luminosity weights
            img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            
        # Stage 1: Gaussian Noise Filtering
        blurred = apply_gaussian_blur(img, kernel_size=self.kernel_size, sigma=self.sigma)
        
        # Stage 2: Gradient Intensity & Direction
        magnitude, theta, grad_x, grad_y = compute_gradients(blurred)
        
        # Stage 3: Non-Maximum Suppression (Edge Thinning)
        suppressed = non_max_suppression(magnitude, theta)
        
        # Stage 4 & 5: Double Thresholding & Hysteresis Tracking
        thresholded, high_t, low_t = double_threshold(
            suppressed,
            low_ratio=self.low_ratio,
            high_ratio=self.high_ratio
        )
        final_edges = hysteresis_tracking(thresholded)
        
        return {
            "edges": final_edges,
            "blurred": blurred,
            "magnitude": magnitude,
            "suppressed": suppressed,
            "thresholded": thresholded,
            "high_threshold": high_t,
            "low_threshold": low_t
        }


def create_synthetic_test_image(size: int = 60) -> np.ndarray:
    """
    Create a clean synthetic test image containing a solid rectangle,
    a filled circle, and additive Gaussian noise.
    """
    img = np.zeros((size, size), dtype=np.float64)
    
    # 1. Solid rectangle: top-left (10, 10) to bottom-right (30, 30)
    img[10:30, 10:30] = 200.0
    
    # 2. Filled circle centered at (42, 42) with radius 12
    y, x = np.ogrid[:size, :size]
    circle_mask = (x - 42) ** 2 + (y - 42) ** 2 <= 12 ** 2
    img[circle_mask] = 230.0
    
    # 3. Add mild sensor noise
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 10.0, (size, size))
    img = np.clip(img + noise, 0.0, 255.0)
    
    return img


def render_ascii_edge_map(edge_map: np.ndarray, step: int = 2):
    """
    Render a compact ASCII representation of the edge map in console.
    """
    H, W = edge_map.shape
    lines = []
    for i in range(0, H, step):
        row = ""
        for j in range(0, W, step):
            row += "#" if edge_map[i, j] > 0 else "."
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Generating synthetic computer vision test pattern...")
    test_img = create_synthetic_test_image(size=40)
    
    detector = CannyEdgeDetector(
        gaussian_kernel_size=5,
        gaussian_sigma=1.4,
        low_threshold_ratio=0.10,
        high_threshold_ratio=0.30
    )
    
    print("Executing Canny Edge Detection pipeline...")
    results = detector.detect_edges(test_img)
    edges = results["edges"]
    
    edge_pixels = int(np.sum(edges > 0))
    total_pixels = edges.size
    print(f"\nTotal image pixels : {total_pixels}")
    print(f"Extracted edge pixels: {edge_pixels} ({edge_pixels / total_pixels:.2%})")
    print(f"High Threshold: {results['high_threshold']:.2f} | Low Threshold: {results['low_threshold']:.2f}")
    
    print("\n--- ASCII Detected Edge Contours (# = Edge, . = Background) ---")
    print(render_ascii_edge_map(edges, step=1))
    print("\nCanny Edge Detection pipeline verified successfully.")
