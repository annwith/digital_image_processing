"""
Application for document aligment.
"""

import argparse
import cv2 as cv
from skimage.transform import rotate
import numpy as np
import math
import matplotlib.pyplot as plt
import pytesseract


def configure_command_line_arguments():
    """
    Configure and retrieve command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command-line 
        arguments.

    This function sets up an ArgumentParser object to parse command-line 
    arguments.
    It adds arguments for the image path and slope precision.
    The function then parses the command-line arguments and returns the parsed
    arguments.
    """

    # Configure the command line arguments parser
    parser = argparse.ArgumentParser(
        description='Load a image and return the slope of the text.')

    # Add an argument for the image path
    parser.add_argument(
        '-i', 
        '--image_path', 
        type=str,
        help='Path to the image file.',
        required=True
    )

    # Add an argument for the precision
    parser.add_argument(
        '-p', 
        '--precision', 
        type=float,
        help='Precision of the slope calculation.',
        required=True
    )

    # Add an argument for the mode
    parser.add_argument(
        '-m', 
        '--mode', 
        type=int,
        help='0 for horizontal projection and 1 for Hough Transform.',
        required=True
    )

    return parser.parse_args()


def objective_function(profile: np.ndarray) -> float:
    """
    Calculates the square of the differences between adjacent cells in the profile.
   
    Parameters:
        profile (np.ndarray): The profile to analyze.
    Returns:
        float: The sum of squares of the differences between adjacent cells.
    
    """
    # Calculate the difference between adjacent cells
    diff = np.diff(profile)

    # Returns the sum of squares of the differences
    return np.sum(diff ** 2)


def slope_from_horizontal_projection(
    image: np.ndarray,
    precision: float = 1
) -> np.ndarray:
    """
    Calculate the slope of the text in the image using horizontal projection.
    
    Parameters:
        image (np.ndarray): The grayscale image to analyze.
        precision (float): The precision of the slope calculation.
    Returns:
        np.ndarray: The possible slopes of the text.

    """
    # Threshold the image - How to make this good for all images?
    # I think local method will work better
    # _, binary_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    # Apply edge detection method on the image
    binary_image = cv.Canny(image, 50, 150) # Como encontrar esses parâmetros?

    # Show the binary image
    cv.imshow('Binary Image', binary_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Initialize the nd.array of values
    func_values = {}

    # Precision decimal places
    precision_decimal_places = int(math.log10(1 / precision))

    # Initialize the start and end angles and the steps
    start = [0]
    end = [360]
    steps = np.geomspace(start=1, stop=precision, num=precision_decimal_places + 1, endpoint=True)
    max_angles = np.array([])
    
    # Calculate the objective function for each angle
    print("******" * 5)
    for step in steps:
        
        # Set the angles to be tested
        angles = np.array([], dtype=float)
        for i in range(len(start)):
            angles = np.append(angles, np.arange(start[i], end[i], step))

        print("Step:", step)
        print("Start:", start)
        print("End:", end - step)

        for angle in angles:

            # Perform the rotation
            rotated_image = rotate(binary_image, angle, resize=True, cval=1)

            # Calculate number of zeros in each line
            profile = np.sum(rotated_image == 0, axis=1)

            # Calculate the value of the objective function
            func_values[angle] = objective_function(profile)

        # Get the angle with the maximum value
        max_value = max(func_values.values())
        max_angles = [key for key, value in func_values.items() if value == max_value]
        max_angles = np.array(max_angles)
        max_angles = np.unique(np.around(max_angles, precision_decimal_places))
        start = max_angles - step + (step / 10)
        end = max_angles + step

        print("Max Angles:", max_angles)
        print("******" * 5)

    return max_angles


def slope_from_hough_transform(
    image: np.ndarray,
    precision: float = 1
) -> np.ndarray:
    """
    Calculate the slope of the text in the image using Hough Transform.
    
    Parameters:
        image (np.ndarray): The grayscale image to analyze.
        precision (float): The precision of the slope calculation.
    Returns:
        np.ndarray: The possible slopes of the text.

    """

    # Apply edge detection method on the image
    binary_image = cv.Canny(image, 50, 150) # How to make this good for all images?

    # Show the binary image
    cv.imshow('Binary Image', binary_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # How to set a good threshold
    # Could be inputed by the user [Optional] Nah, it's not a good idea
    # Could be calculated based on the size of the image or the number of black pixels
    # Could be found iteratively [The most efficient way, but most complex]

    # Perform Hough Line Transform
    lines = cv.HoughLines(
        image=binary_image, # Input image
        rho=1, # Distance resolution in pixels
        theta=np.pi / (180 / precision), # Angle resolution in radians
        threshold=(binary_image.shape[0] + binary_image.shape[1]) // 6 # Depends of the size of the image... [Find a good value]
    )

    # Checks if any line was found
    if lines is None:
        assert False, "No lines were found."

    # Print the number of lines found
    print(f'Number of lines found: {len(lines)}')

    # Get angles of the lines
    angles = np.array([l[0][1] for l in lines])

    # Convert angles to degrees and shift them by 90 degrees
    angles = angles * 180 / np.pi + 90

    # Median
    median = np.median(angles)
    print(f'Median: {median}')

    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    # h, theta, d = hough_line(image, theta=tested_angles)

    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr

        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a*r

        # y0 stores the value rsin(theta)
        y0 = b*r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        cv.line(binary_image, (x1, y1), (x2, y2), 127, 2)
    
    # Show the Hough Lines
    cv.imshow('Hough Lines', binary_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return np.array([round(median, int(math.log10(1 / precision)))])


# Configure the command line arguments
args = configure_command_line_arguments()

# Get image name and the precision
image_name = args.image_path.split('/')[-1].split('.')[0]
precision = args.precision

# Load image
image = cv.imread(args.image_path, cv.IMREAD_GRAYSCALE)

# Assert that the image was read
assert image is not None, "File could not be read."

if args.mode == 0:
    # Calculate the slope of the text
    slopes = slope_from_horizontal_projection(
        image=image,
        precision=precision
    )
elif args.mode == 1:
    # Calculate the slope of the text
    slopes = slope_from_hough_transform(
        image=image,
        precision=precision
    )
else:
    raise ValueError('Invalid mode.')

# Get the slopes + 180 degrees
slopes = np.append(slopes, (slopes + 180) % 360)

# Best median confidence
best_median_confidence = -float('inf')
best_slope = None

for s in slopes:
    # Print the slope
    print(f'Possible slope: {s}')

    # Rotate the image
    rotated_image = rotate(image, s, resize=True, cval=1)

    # Convert the rotated image to uint8
    rotated_image = np.array(rotated_image * 255, dtype=np.uint8)

    # Use Tesseract to get detailed information, including confidence levels
    data = pytesseract.image_to_data(rotated_image, output_type=pytesseract.Output.DICT)

    # Print mean confidence and median confidence
    print(f"Median Confidence: {np.median(data['conf'])}")
    print("******" * 5)

    # Update the best median confidence
    if np.median(data['conf']) > best_median_confidence:
        best_median_confidence = np.median(data['conf'])
        best_slope = s

# Print the best slope
print(f'Best slope: {best_slope}')

# Rotate the image
rotated_image = rotate(image, best_slope, resize=True, cval=1)

# Convert the rotated image to uint8
rotated_image = np.array(rotated_image * 255, dtype=np.uint8)

# Show the rotated image
cv.imshow('Rotated Image', rotated_image)
cv.waitKey(0)
cv.destroyAllWindows()

# Define decimal places
decimal = - int(math.log10(precision))

# Define the output path and decimal places
output_path_image = 'output_images/{image_name}_mode_{mode}_rotated_{slope:.{decimal}f}.png'
output_path_image = output_path_image.replace('{decimal}', str(decimal))

# Print the output path
print("Rotated image saved in:", output_path_image.format(
    image_name=image_name, 
    mode=args.mode, slope=s
))

# Save the rotated image
cv.imwrite(output_path_image.format(
    image_name=image_name, 
    mode=args.mode, slope=s
), rotated_image)

# Perform text extraction using Tesseract
text = pytesseract.image_to_string(rotated_image)

# Define the output path and decimal places
output_path_text = 'output_texts/{image_name}_mode_{mode}_rotated_{slope:.{decimal}f}.txt'
output_path_text = output_path_text.replace('{decimal}', str(decimal))

# Print the output path
print("Rotated image text saved in:", output_path_text.format(
    image_name=image_name, 
    mode=args.mode, slope=s
))

# Save the rotated image text
with open(output_path_text.format(
    image_name=image_name, 
    mode=args.mode, slope=s
), 'w') as file:
    file.write(text)