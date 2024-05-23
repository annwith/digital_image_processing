"""
Application for document aligment using horizontal projection.

O angulo escolhido ́e aquele que otimiza uma função objetivo calculada sobre o 
perfil da projeção horizontal. Um exemplo de função objetivo é a soma dos 
quadrados das diferenças dos valores em células adjacentes do perfil de projeção.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def objective_function(profile: np.ndarray) -> float:
    """
    Calculates the square of the differences between adjacent cells in the profile.
    :param profile: Profile of the horizontal projection
    :return: Value of the objective function
    """
    # Calculate the difference between adjacent cells
    diff = np.diff(profile)

    # Calculate the sum of squares of the differences
    value = np.sum(diff ** 2)

    return value


def slope_from_horizontal_projection(
    binary_image: np.ndarray,
    angles: np.ndarray = np.arange(0, 360, 1)
) -> float:
    """
    Calculate the slope of the text in the image using horizontal projection.
    :param binary_image: Binary image
    :return: Slope of the text
    """

    # Get the dimensions of the image
    (h, w) = binary_image.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Initialize the list of values
    values = []

    # Calculate the objective function for each angle
    for angle in angles:
        
        # Get the rotation matrix
        M = cv.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        rotated_image = cv.warpAffine(image, M, (w, h))

        # Show the rotated image
        cv.imshow('Rotated Image', rotated_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Calculate number of zeros in each line
        profile = np.sum(rotated_image == 0, axis=1)
        
        # Calculate the value of the objective function
        values.append(objective_function(profile))

    # Get the angle with the maximum value
    slope = angles[np.argmax(values)]

    return slope

# Read the image
image = cv.imread('input_images/neg_4.png', cv.IMREAD_GRAYSCALE)

# Threshold the image
_, bin_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

# Calculate number of zeros in each line
zero_count = np.sum(bin_image == 0, axis=1)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the image
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax1.set_aspect('auto')
ax1.axis('off')

# Plot the zero count
ax2.plot(zero_count, range(len(zero_count)))
ax2.set_title('Zero Count')
ax2.set_xlabel('Count')
ax2.set_ylim([0, image.shape[0]])
ax2.set_aspect('auto')
ax2.axis('off')
ax2.invert_yaxis()  # Invert y-axis to match image orientation

plt.show()