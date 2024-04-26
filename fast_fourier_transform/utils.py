"""
Auxiliary functions.
"""

import numpy as np


def create_mask(
    height: int,
    width: int,
    radius: float,
    high_pass: bool = False
) -> np.ndarray:
    """
    Creates a mask for a low-pass or high-pass filter.

    Args:
        height: Height of the image.
        width: Width of the image.
        radius: Radius of the filter.
        high_pass: If True, creates a high-pass filter. Otherwise, creates a 
            low-pass filter.
    Returns:
        Filter mask.
    """

    # Calculates the image center
    center = (height // 2, width // 2)

    # Creates two grids of coordinates
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Calculates the distances of each pixel to the center
    distances = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)

    # Creates the mask
    if high_pass:
        mask = distances > radius
    else:
        mask = distances <= radius

    return mask
