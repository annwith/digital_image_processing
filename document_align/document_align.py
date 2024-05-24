"""
Application for document aligment.
"""

import argparse
import cv2 as cv
from skimage.transform import rotate
import numpy as np
import math
import pytesseract
import time


def is_power_of_ten(x: float) -> bool:
    """
    Check if a number is a power of ten smaller than 10.

    Parameters:
        x (float): The number to check.
    Returns:
        bool: True if the number is a power of ten, False otherwise.
    """
    return x > 0 and x < 10 and 10**round(math.log10(x)) == x


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

    # Print information
    print("******" * 6)
    print("Aligment with horizontal projection.")

    # Apply edge detection method on the image
    binary_image = cv.Canny(image, 50, 150)

    # Start the timer
    start_time = time.time()

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
    print("******" * 6)
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

        print("Max angles:", max_angles)
        print("******" * 6)

    # Stop the timer
    end_time = time.time()

    # Print the total time taken
    print(f"Horizontal projection time: {end_time - start_time:.2f} seconds")
    print("******" * 6)

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

    # Print information
    print("******" * 6)
    print("Aligment with hough transform.")

    # Apply edge detection method on the image
    binary_image = cv.Canny(image, 50, 150)

    # Show the binary image
    cv.imshow('Binary Image', binary_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Start the timer
    start_time = time.time()

    # Perform Hough Line Transform
    lines = cv.HoughLines(
        image=binary_image, # Input image
        rho=1, # Distance resolution in pixels
        theta=np.pi / (180 / precision), # Angle resolution in radians
        threshold=(binary_image.shape[0] + binary_image.shape[1]) // 6
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

    # Print the median
    print(f'Median: {median}')
    print("******" * 6)

    # Stop the timer
    end_time = time.time()

    # Print the total time taken
    print(f"cv.HoughLines time: {end_time - start_time:.2f} seconds")
    print("******" * 6)

    # Show the detected lines
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*r
        y0 = b*r

        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))

        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv.line(binary_image, (x1, y1), (x2, y2), 127, 2)
    
    # Show the Hough Lines
    cv.imshow('Hough Lines', binary_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return np.array([round(median, int(math.log10(1 / precision)))])


def find_best_slope_using_ocr(
    slopes: np.ndarray,
) -> float:
    """
    Gets the correct slope of the text using Tesseract OCR. Runs OCR to each 
    possible slope, gets the confidente for each detected word and calculates 
    the median confidence. The slope with the highest median confidence is
    returned.

    If there is a tie, the slope with the lowest angle is returned (angles are
    on the [0, 360[ interval).
    
    Parameters:
        slopes (np.ndarray): The possible slopes of the text.    
    
    Returns:
        float: The slope of the text.

    """
    # Print information
    print("Find correct slope using OCR.")

    # Get the slopes + 180 degrees
    slopes = np.append(slopes, (slopes + 180) % 360)

    # Sort the slopes
    slopes = np.sort(slopes)

    # Best median confidence
    best_median_confidence = -float('inf')
    best_slope = None

    print("******" * 6)
    for s in slopes:
        # Print the slope
        print(f'Slope: {s}')

        # Rotate the image
        rotated_image = rotate(image, s, resize=True, cval=1)

        # Convert the rotated image to uint8
        rotated_image = np.array(rotated_image * 255, dtype=np.uint8)

        # Use Tesseract to get detailed information, including confidence levels
        data = pytesseract.image_to_data(rotated_image, output_type=pytesseract.Output.DICT)

        # Print mean confidence and median confidence
        print(f"Median confidence: {np.median(data['conf'])}")
        print("******" * 6)

        # Update the best median confidence
        if np.median(data['conf']) > best_median_confidence:
            best_median_confidence = np.median(data['conf'])
            best_slope = s

    return best_slope


# Configure the command line arguments
args = configure_command_line_arguments()

# Get image name and the precision
image_name = args.image_path.split('/')[-1].split('.')[0]
precision = args.precision

# Assert precision is a power of ten
assert is_power_of_ten(precision), "Precision must be a power of ten between 0 and 1."

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

# Find the best slope using OCR
best_slope = find_best_slope_using_ocr(slopes=slopes)

# Print the best slope
print(f'Correct slope: {best_slope}')


# Rotate the image
rotated_image = rotate(image, best_slope, resize=True, cval=1)

# Convert the rotated image to uint8
rotated_image = np.array(rotated_image * 255, dtype=np.uint8)

# Define decimal places
decimal = - int(math.log10(precision))

# Define the output path and decimal places
output_path_image = 'output_images/{image_name}_mode_{mode}_rotated_{slope:.{decimal}f}.png'
output_path_image = output_path_image.replace('{decimal}', str(decimal))

# Print the output path
print("Rotated image saved in:", output_path_image.format(
    image_name=image_name, 
    mode=args.mode, slope=best_slope
))

# Save the rotated image
cv.imwrite(output_path_image.format(
    image_name=image_name, 
    mode=args.mode, slope=best_slope
), rotated_image)


# Perform text extraction using Tesseract
text = pytesseract.image_to_string(image)

# Define the output path and decimal places
output_path_text = 'output_texts/{image_name}.txt'

# Print the output path
print("Image text saved in:", output_path_text.format(
    image_name=image_name, 
    mode=args.mode, slope=best_slope
))

# Save the rotated image text
with open(output_path_text.format(
    image_name=image_name, 
    mode=args.mode, slope=best_slope
), 'w') as file:
    file.write(text)


# Perform text extraction using Tesseract
text = pytesseract.image_to_string(rotated_image)

# Define the output path and decimal places
output_path_text = 'output_texts/{image_name}_mode_{mode}_rotated_{slope:.{decimal}f}.txt'
output_path_text = output_path_text.replace('{decimal}', str(decimal))

# Print the output path
print("Rotated image text saved in:", output_path_text.format(
    image_name=image_name, 
    mode=args.mode, slope=best_slope
))

# Save the rotated image text
with open(output_path_text.format(
    image_name=image_name, 
    mode=args.mode, slope=best_slope
), 'w') as file:
    file.write(text)