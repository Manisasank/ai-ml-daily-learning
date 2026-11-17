"""
Non-Maximum Suppression (NMS) Module
Thins blurred edge response ridges into 1-pixel wide edge candidates
by suppressing non-peak gradient responses along the local gradient vector.
"""

import numpy as np


def non_max_suppression(magnitude: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Apply directional non-maximum suppression along gradient normals.
    
    Parameters:
        magnitude: 2D array of gradient intensities (H, W).
        theta: 2D array of gradient angles in degrees [0, 180).
        
    Returns:
        Suppressed 2D array (H, W) where only local directional maxima are preserved.
    """
    H, W = magnitude.shape
    suppressed = np.zeros((H, W), dtype=np.float64)
    
    # Iterate across interior pixels (borders zeroed)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            angle = theta[i, j]
            mag = magnitude[i, j]
            
            # Target neighbors along gradient direction
            q = 255.0
            r = 255.0
            
            # 0 degrees: East-West gradient (vertical edge) -> compare with East and West
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
                
            # 45 degrees: Northeast-Southwest gradient -> compare with NE and SW
            elif 22.5 <= angle < 67.5:
                q = magnitude[i - 1, j + 1]
                r = magnitude[i + 1, j - 1]
                
            # 90 degrees: North-South gradient (horizontal edge) -> compare with North and South
            elif 67.5 <= angle < 112.5:
                q = magnitude[i - 1, j]
                r = magnitude[i + 1, j]
                
            # 135 degrees: Northwest-Southeast gradient -> compare with NW and SE
            elif 112.5 <= angle < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
                
            # Keep if pixel intensity is strictly higher than both directional neighbors
            if (mag >= q) and (mag >= r):
                suppressed[i, j] = mag
            else:
                suppressed[i, j] = 0.0
                
    return suppressed


if __name__ == "__main__":
    # Create synthetic gradient ramp (thick 3-pixel edge)
    test_mag = np.zeros((7, 7))
    test_mag[:, 2] = 100.0
    test_mag[:, 3] = 200.0  # True peak
    test_mag[:, 4] = 100.0
    
    test_theta = np.zeros((7, 7))  # 0 degrees (horizontal gradient)
    
    thin = non_max_suppression(test_mag, test_theta)
    print("Original multi-pixel ridge middle row:")
    print(test_mag[3, :])
    print("\nAfter Non-Maximum Suppression (thinned to 1-pixel peak):")
    print(thin[3, :])
    assert thin[3, 3] == 200.0 and thin[3, 2] == 0.0 and thin[3, 4] == 0.0
    print("NMS unit test passed successfully.")
