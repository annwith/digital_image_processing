"""
Application for document aligment.
"""

import argparse
import cv2 as cv
from skimage.transform import rotate
import numpy as np
import math


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
    :param profile: Profile of the horizontal projection
    :return: Value of the objective function
    """
    # Calculate the difference between adjacent cells
    diff = np.diff(profile)

    # Returns the sum of squares of the differences
    return np.sum(diff ** 2)


# Improve the time of this one
def slope_from_horizontal_projection(
    image: np.ndarray,
    precision: float = 1
) -> np.ndarray:
    """
    Calculate the slope of the text in the image using horizontal projection.
    :param image: Image
    :return: List of possible slopes of the text
    """
    # Threshold the image - How to make this good for all images?
    # I think local method will work better
    _, binary_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    # Apply edge detection method on the image
    binary_image = cv.Canny(image, 50, 150)

    # Show the binary image
    cv.imshow('Binary Image', binary_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Set the angles to be tested
    angles = np.arange(0, 180, precision)

    # Initialize the nd.array of values
    values = np.array([], dtype=int)

    # Calculate the objective function for each angle
    for angle in angles:

        # Perform the rotation
        rotated_image = rotate(binary_image, angle, resize=True, cval=1)

        # Calculate number of zeros in each line
        profile = np.sum(rotated_image == 0, axis=1)

        # Calculate the value of the objective function
        values = np.append(values, objective_function(profile))

    # Returns the angles with the maximum value
    return np.where(values == values.max())[0] / (len(angles) / 180)


def slope_from_hough_transform(
    image: np.ndarray,
    precision: float = 1
) -> np.ndarray:
    """
    Calculate the slope of the text in the image using Hough Transform.
    :param binary_image: Binary image
    :return: List of possible slopes of the text
    """

    # Apply edge detection method on the image
    binary_image = cv.Canny(image, 50, 150) # How to make this good for all images?

    # Show the binary image
    cv.imshow('Binary Image', binary_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # How to set a good threshold
    # Could be inputed by the user [Optional]
    # Could be calculated based on the size of the image or the number of black pixels
    # Could be found iteratively

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
    print("Angles:", angles)

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

    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    # h, theta, d = hough_line(image, theta=tested_angles)

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
slopes = np.append(slopes, (slopes + 180))

for s in slopes:
    # Print the slope
    print(f'Possible slope: {s}')

    # Rotate the image
    rotated_image = rotate(image, s, resize=True)

    # Show the rotated image
    cv.imshow('Rotated Image', rotated_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Define decimal places
    decimal = - int(math.log10(precision))

    # Define the output path and decimal places
    output_path_txt = 'output_images/{image_name}_mode_{mode}_rotated_{slope:.{decimal}f}.png'
    output_path_txt = output_path_txt.replace('{decimal}', str(decimal))

    # Save the rotated image
    cv.imwrite(output_path_txt.format(
        image_name=image_name, 
        mode=args.mode, slope=s
    ), rotated_image)