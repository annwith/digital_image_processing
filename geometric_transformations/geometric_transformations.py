"""
Geometric transformations on an image.
"""
import argparse
import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """
    Function to define command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Geometric transformations on an image.")

    parser.add_argument(
        '-a', 
        '--angle', 
        type=float,
        default=0,
        help="Rotation angle in degrees (counterclockwise).")

    parser.add_argument(
        '-s', 
        '--scale', 
        type=float,
        default=1.0,
        help="Scaling factor.")

    parser.add_argument(
        '-d', 
        '--dimension', 
        nargs=2,
        type=int,
        help="Output image dimension (height width).")

    parser.add_argument(
        '-m', 
        '--method', 
        choices=['nearest', 'bilinear', 'bicubic', 'lagrange'],
        default='nearest',
        help="Interpolation method.")

    parser.add_argument(
        '-i', 
        '--input', 
        required=True,
        help="Input image in PNG format.")

    parser.add_argument(
        '-o', 
        '--output', 
        required=True,
        help="Output image in PNG format.")

    return parser.parse_args()


def plot_histogram(image, title):
    """Plot the histogram of an image."""
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()  # Normalize to sum to 1
    hist *= 100  # Convert to percentage

    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Percentage")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.ylim([0, 100])  # Set y-axis limit to 100%
    plt.show()


def translation_matrix(tx, ty):
    """
    Create a translation matrix.

    Parameters:
        tx (int): Translation along x-axis.
        ty (int): Translation along y-axis.
    Returns:
        numpy.ndarray: Translation matrix.
    """
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])


def scale_matrix(sx, sy):
    """
    Create a scaling matrix.

    Parameters:
        sx (float): Scaling factor along x-axis.
        sy (float): Scaling factor along y-axis.
    Returns:
        numpy.ndarray: Scaling matrix.
    """
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])


def rotation_matrix(angle):
    """
    Create a 2D rotation matrix in homogeneous coordinates.

    Parameters:
        angle (float): Rotation angle in degrees (counterclockwise).
        
    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    angle_rad = np.deg2rad(angle)  # Convert angle from degrees to radians

    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])


def nearest_neighbor_interpolation(image, x, y):
    """
    Nearest neighbor interpolation.

    Parameters:
        image (numpy.ndarray): Input image.
        x (float): x-coordinate.
        y (float): y-coordinate.
    Returns:
        numpy.ndarray: Interpolated pixel value.
    """
    h, w = image.shape[0], image.shape[1]
    channels = None
    x, y = round(x), round(y)

    if len(image.shape) == 3:
        channels = image.shape[2]

    if x < 0 or x >= w or y < 0 or y >= h:
        if channels:
            return np.array([0] * channels)
        return 0

    return image[y, x]


def bilinear_interpolation(image, x, y):
    """
    Bilinear interpolation.

    Parameters:
        image (numpy.ndarray): Input image.
        x (float): x-coordinate.
        y (float): y-coordinate.
    Returns:
        numpy.ndarray: Interpolated pixel value.
    """
    h, w = image.shape[0], image.shape[1]
    channels = None
    x1, y1 = int(np.floor(x)), int(np.floor(y))

    dx = x - x1
    dy = y - y1

    if x1 < 0 or (x1 + 1) >= w or y1 < 0 or (y1 + 1) >= h:
        if channels:
            return np.array([0] * channels)
        return 0

    p11 = image[y1, x1]
    p12 = image[y1 + 1, x1]
    p21 = image[y1, x1 + 1]
    p22 = image[y1 + 1, x1 + 1]

    p = (1 - dx) * (1 - dy) * p11 + \
            dx * (1 - dy) * p21 + \
            (1 - dx) * dy * p12 + \
            dx * dy * p22

    # Clip the pixel value
    p = np.clip(p, 0, 255)

    # Round the pixel value
    p = np.round(p)

    return p


def P(t):
    """
    Helper function for cubic B-spline.

    Parameters:
        t (float): Input value.
    Returns:
        float: Output value.
    """
    return t if t > 0 else 0


def B_spline_cubic(s):
    """
    Cubic B-spline function.

    Parameters:
        s (float): Input value.
    Returns:
        float: Output value.
    """
    return 1/6 * (
        P(s + 2) ** 3 - \
        4 * P(s + 1) ** 3 + \
        6 * P(s) ** 3 - \
        4 * P(s - 1) ** 3
    )


def bicubic_interpolation(image, x, y):
    """
    Bicubic interpolation.

    Parameters:
        image (numpy.ndarray): Input image.
        x (float): x-coordinate.
        y (float): y-coordinate.
    Returns:
        numpy.ndarray: Interpolated pixel value.
    """
    h, w = image.shape[0], image.shape[1]
    channels = None

    _x, _y = int(np.floor(x)), int(np.floor(y))

    dx = x - _x
    dy = y - _y

    if _x - 1 < 0 or _x + 2 >= w or _y -1 < 0 or _y + 2 >= h:
        if channels:
            return np.array([0] * channels)
        return 0

    if channels:
        p = np.zeros(channels)
    else:
        p = 0

    for m in range(-1, 3):
        for n in range(-1, 3):
            p += image[_y + n, _x + m] * B_spline_cubic(m - dx) * B_spline_cubic(dy - n)

    # Clip the pixel value
    p = np.clip(p, 0, 255)

    # Round the pixel value
    p = np.round(p)

    return p


def L(image, dx, x, y, n):
    """
    Helper function for lagrange interpolation.
    Calculate the value of L for a given image, dx, x, y, and n.

    Parameters:
    image (numpy.ndarray): The input image.
    dx (int): The value of dx.
    x (int): The x-coordinate.
    y (int): The y-coordinate.
    n (int): The value of n.

    Returns:
    float: The calculated value of L.
    """
    return (
        -dx * (dx-1) * (dx-2) * image[y+n-2, x-1] / 6 + \
        (dx+1) * (dx-1) * (dx-2) * image[y+n-2, x] / 2 + \
        -dx * (dx+1) * (dx-2) * image[y+n-2, x+1] / 2 + \
        dx * (dx+1) * (dx-1) * image[y+n-2, x+2] / 6
    )


def lagrange_interpolation(image, x, y):
    """
    Lagrange interpolation.

    Parameters:
        image (numpy.ndarray): Input image.
        x (float): x-coordinate.
        y (float): y-coordinate.
    Returns:
        numpy.ndarray: Interpolated pixel value.
    """
    h, w = image.shape[0], image.shape[1]
    channels = None

    _x, _y = int(np.floor(x)), int(np.floor(y))

    dx = x - _x
    dy = y - _y

    if _x - 1 < 0 or _x + 2 >= w or _y -1 < 0 or _y + 2 >= h:
        if channels:
            return np.array([0] * channels)
        return 0

    p = -dy * (dy-1) * (dy-2) * L(image, dx, _x, _y, 1) / 6 + \
        (dy+1) * (dy-1) * (dy-2) * L(image, dx, _x, _y, 2) / 2 + \
        -dy * (dy+1) * (dy-2) * L(image, dx, _x, _y, 3) / 2 + \
        dy * (dy+1) * (dy-1) * L(image, dx, _x, _y, 4) / 6

    # Clip the pixel value
    p = np.clip(p, 0, 255)

    # Round the pixel value
    p = np.round(p)

    return p


def rotate_image(image, angle, interpolation='nearest'):
    """
    Rotate an image by a given angle.

    Parameters:
        image (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees (counterclockwise).
        interpolation (str): Interpolation method.
    Returns:
        numpy.ndarray: Rotated image.
    """

    # Get image dimensions
    if len(image.shape) == 2:
        h, w = image.shape
    elif len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        raise ValueError("Invalid image shape.")

    # Find the center of the image
    center = np.array([h // 2, w // 2])

    # Rotation matrix
    R = rotation_matrix(angle)

    # Translation matrixes
    T_to_center = translation_matrix(-center[1], -center[0])
    T_back = translation_matrix(center[1], center[0])

    # Translate to the center
    T_sequence = T_back @ R @ T_to_center

    print("Transformation matrix:\n", T_sequence)

    # Create the output image
    rotated_image = np.zeros_like(image)

    # Start the timer
    start = time.time()

    for y in range(h):
        for x in range(w):
            # Define coordinates
            coord = np.array([x, y, 1])

            # Translate back
            new_coord = T_sequence @ coord

            # Image coordinates
            new_x, new_y = new_coord[:2]

            # Interpolation
            if interpolation == 'nearest':
                rotated_image[y, x] = nearest_neighbor_interpolation(image, new_x, new_y)
            elif interpolation == 'bilinear':
                rotated_image[y, x] = bilinear_interpolation(image, new_x, new_y)
            elif interpolation == 'bicubic':
                rotated_image[y, x] = bicubic_interpolation(image, new_x, new_y)
            elif interpolation == 'lagrange':
                rotated_image[y, x] = lagrange_interpolation(image, new_x, new_y)

    # End the timer
    end = time.time()

    # Print the elapsed time
    elapsed_time = end - start
    print(f"Time taken to construct the new image: {elapsed_time:.4f} seconds")

    return rotated_image


def scale_image(image, scale, dimensions, interpolation='nearest'):
    """
    Scale an image by a given factor or resize to given dimensions.

    Parameters:
        image (numpy.ndarray): Input image.
        scale (float): Scaling factor.
        dimensions (tuple): Output image dimensions (width, height).
        interpolation (str): Interpolation method.
    Returns:
        numpy.ndarray: Scaled image.
    """
    # Get image dimensions
    if len(image.shape) == 2:
        h, w = image.shape
        c = 1
    elif len(image.shape) == 3:
        h, w, c = image.shape
    else:
        raise ValueError("Invalid image shape.")

    # Get the scaling factors
    if dimensions:
        height, width = dimensions
        scale_x = width / w
        scale_y = height / h
    else:
        scale_x = scale
        scale_y = scale

    # Scaling matrix
    S = scale_matrix(1/scale_x, 1/scale_y)

    # Print the transformation matrix
    print("Transformation matrix:\n", S)

    # Create the output image
    new_h = round(h * scale_y)
    new_w = round(w * scale_x)

    if c == 1:
        scaled_image = np.zeros((new_h, new_w))
    else:
        scaled_image = np.zeros((new_h, new_w, c))

    # Start the timer
    start = time.time()

    for y in range(new_h):
        for x in range(new_w):
            # Define coordinates
            coord = np.array([x, y, 1])

            # Scale coordinates
            coord = S @ coord

            # Translate back to image coordinates
            new_x, new_y = coord[:2]

            # Interpolation
            if interpolation == 'nearest':
                scaled_image[y, x] = nearest_neighbor_interpolation(image, new_x, new_y)
            elif interpolation == 'bilinear':
                scaled_image[y, x] = bilinear_interpolation(image, new_x, new_y)
            elif interpolation == 'bicubic':
                scaled_image[y, x] = bicubic_interpolation(image, new_x, new_y)
            elif interpolation == 'lagrange':
                scaled_image[y, x] = lagrange_interpolation(image, new_x, new_y)

    # End the timer
    end = time.time()

    # Print the elapsed time
    elapsed_time = end - start
    print(f"Time taken to construct the new image: {elapsed_time:.4f} seconds")

    return scaled_image


def transform_image(args):
    # Read the input image
    image = cv.imread(args.input, cv.IMREAD_UNCHANGED)

    # Set interpolation flags
    flags_dict = {
        'nearest': cv.INTER_NEAREST,
        'bilinear': cv.INTER_LINEAR,
        'bicubic': cv.INTER_CUBIC,
        'lagrange': cv.INTER_CUBIC # Lagrange interpolation is not implemented in OpenCV
    }

    if image is None:
        raise FileNotFoundError(f"Input file {args.input} not found.")

    if args.scale != 1.0 and args.dimension is not None:
        raise ValueError("Set only one scaling argument (scale or dimension).")

    if (args.scale != 1.0 and args.angle) or (args.dimension and args.angle):
        raise ValueError("Rotation and scaling cannot be done simultaneously.")

    # Apply rotation
    if args.angle != 0:
        # Apply custom rotation
        rotated_image = rotate_image(image, args.angle, interpolation=args.method)

        # Apply OpenCV rotation
        h, w = image.shape[0], image.shape[1]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, args.angle, 1.0)
        rotated_image_cv = cv.warpAffine(image, M, (w, h), flags=flags_dict[args.method])

        # Compare with OpenCV rotation (Both images and subtraction of them)
        rotated_diff_image = cv.absdiff(rotated_image, rotated_image_cv)

        # Print the maximum difference
        print("Max difference:", np.max(rotated_diff_image))
        print("Median difference:", np.median(rotated_diff_image))

        # Display the images
        cv.imshow('Original Image', image)
        cv.imshow('Custom Rotated Image', rotated_image)
        cv.imshow('OpenCV Rotated Image', rotated_image_cv)
        cv.imshow('Difference Image', rotated_diff_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Plot the difference histogram
        plot_histogram(rotated_diff_image, "Difference Histogram")

    # Apply scaling
    if args.scale != 1.0:
        # Apply custom scaling
        scaled_image = scale_image(image, args.scale, None, interpolation=args.method)

        print(scaled_image.shape)

        # Apply OpenCV scaling
        scaled_image_cv = cv.resize(image, None, fx=args.scale, fy=args.scale, interpolation=flags_dict[args.method])

        print(scaled_image_cv.shape)

        # Compare with OpenCV scaling (Both images and subtraction of them)
        scaled_image = scaled_image.astype(np.uint8)
        scaled_image_cv = scaled_image_cv.astype(np.uint8)
        scaled_diff_image = cv.absdiff(scaled_image, scaled_image_cv)

        print("Max difference:", np.max(scaled_diff_image))        

        # Display the images
        cv.imshow('Original Image', image)
        cv.imshow('Custom Scaled Image', scaled_image)
        cv.imshow('OpenCV Scaled Image', scaled_image_cv)
        cv.imshow('Difference Image', scaled_diff_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Plot the difference histogram
        plot_histogram(scaled_diff_image, "Difference Histogram")
    
    # Resize to specified dimensions
    if args.dimension:
        # Apply custom resizing
        scaled_image = scale_image(image, None, args.dimension, interpolation=args.method)
        scaled_image = scaled_image.astype(np.uint8)

        # Apply OpenCV resizing
        height, width = args.dimension
        scaled_image_cv = cv.resize(image, (width, height), interpolation=flags_dict[args.method])

        # Compare with OpenCV resizing (Both images and subtraction of them)
        scaled_diff_image = cv.absdiff(scaled_image, scaled_image_cv)

        # Display the images
        cv.imshow('Original Image', image)
        cv.imshow('Custom Resized Image', scaled_image)
        cv.imshow('OpenCV Resized Image', scaled_image_cv)
        cv.imshow('Difference Image', scaled_diff_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Plot the difference histogram
        plot_histogram(scaled_diff_image, "Difference Histogram")
    
    # Save the output image
    cv.imwrite(args.output, image)

if __name__ == '__main__':
    args = parse_args()
    transform_image(args)
