"""
Double Thresholding and Hysteresis Edge Connectivity Tracking
Separates candidate pixels into strong, weak, and non-edges, then tracks
8-connected components to preserve genuine contours and discard noise.
"""

from typing import Tuple
import numpy as np


# Label definitions for thresholding states
STRONG_PIXEL = 255
WEAK_PIXEL = 75
NON_EDGE = 0


def double_threshold(
    image: np.ndarray,
    low_ratio: float = 0.08,
    high_ratio: float = 0.22
) -> Tuple[np.ndarray, float, float]:
    """
    Classify pixels into strong, weak, and non-edge categories based on intensity bounds.
    
    Parameters:
        image: Non-maximum suppressed gradient array (H, W).
        low_ratio: Lower bound ratio of the maximum intensity.
        high_ratio: Upper bound ratio of the maximum intensity.
        
    Returns:
        thresholded: 2D array labeled with STRONG_PIXEL, WEAK_PIXEL, or NON_EDGE.
        high_threshold: Absolute calculated high cutoff.
        low_threshold: Absolute calculated low cutoff.
    """
    max_val = np.max(image)
    high_threshold = max_val * high_ratio
    low_threshold = high_threshold * low_ratio
    
    thresholded = np.zeros_like(image, dtype=np.uint8)
    
    # Pixel masks
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
    
    thresholded[strong_i, strong_j] = STRONG_PIXEL
    thresholded[weak_i, weak_j] = WEAK_PIXEL
    
    return thresholded, high_threshold, low_threshold


def hysteresis_tracking(image: np.ndarray) -> np.ndarray:
    """
    Traverse 8-connected neighborhoods around weak edge pixels.
    If a weak pixel is connected to any strong edge pixel, it is promoted to strong;
    otherwise, it is discarded as noise.
    """
    H, W = image.shape
    result = image.copy()
    
    # 8-connectivity neighborhood offsets
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]
    
    # Scan through interior pixels
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if result[i, j] == WEAK_PIXEL:
                # Check if connected to at least one strong edge pixel
                has_strong_neighbor = any(
                    result[i + di, j + dj] == STRONG_PIXEL
                    for di, dj in neighbors
                )
                
                if has_strong_neighbor:
                    result[i, j] = STRONG_PIXEL
                else:
                    result[i, j] = NON_EDGE
                    
    # Clean any residual isolated weak edges
    result[result == WEAK_PIXEL] = NON_EDGE
    return result


if __name__ == "__main__":
    # Test hysteresis on a continuous line with a weak middle link vs isolated noise
    test_cand = np.zeros((7, 7), dtype=np.float64)
    # Connected line: (3, 1)=Strong, (3, 2)=Weak, (3, 3)=Strong
    test_cand[3, 1] = 200.0  # Strong
    test_cand[3, 2] = 50.0   # Weak
    test_cand[3, 3] = 200.0  # Strong
    # Isolated noise
    test_cand[1, 5] = 50.0   # Weak isolated
    
    thresh, high_t, low_t = double_threshold(test_cand, low_ratio=0.1, high_ratio=0.5)
    print(f"High Threshold: {high_t}, Low Threshold: {low_t}")
    print("Thresholded line segment (3, 1..3):", thresh[3, 1:4])
    print("Isolated noise at (1, 5):", thresh[1, 5])
    
    final_edges = hysteresis_tracking(thresh)
    print("After Hysteresis line segment:", final_edges[3, 1:4])
    print("After Hysteresis isolated noise:", final_edges[1, 5])
    
    assert final_edges[3, 2] == STRONG_PIXEL, "Connected weak pixel should be promoted to strong."
    assert final_edges[1, 5] == NON_EDGE, "Isolated weak pixel should be discarded."
    print("Hysteresis edge connectivity test passed successfully.")
