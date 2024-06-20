"""
Geometric transformations on an image.
"""
import argparse
import cv2 as cv
import numpy as np


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
        '-e', 
        '--scale', 
        type=float,
        default=1.0,
        help="Scaling factor.")

    parser.add_argument(
        '-d', 
        '--dimension', 
        nargs=2,
        type=int,
        help="Output image dimension (width height).")

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
    Create a rotation matrix.

    Parameters:
        angle (float): Rotation angle in degrees (counterclockwise).
    Returns:
        numpy.ndarray: Rotation matrix.
    """
    angle_rad = np.deg2rad(-angle)
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
    x, y = round(x), round(y)

    if len(image.shape) == 2:
        channels = 1
    else:
        channels = image.shape[2]

    if x < 0 or x >= w or y < 0 or y >= h:
        return np.array([0] * channels)

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
    x1, y1 = int(np.floor(x)), int(np.floor(y))

    dx = x - x1
    dy = y - y1

    if len(image.shape) == 2:
        channels = 1
    else:
        channels = image.shape[2]

    if x1 < 0 or (x1 + 1) >= w or y1 < 0 or (y1 + 1) >= h:
        return np.array([0] * channels)

    p11 = image[y1, x1]
    p12 = image[y1 + 1, x1]
    p21 = image[y1, x1 + 1]
    p22 = image[y1 + 1, x1 + 1]

    p = (1 - dx) * (1 - dy) * p11 + \
            dx * (1 - dy) * p21 + \
            (1 - dx) * dy * p12 + \
            dx * dy * p22

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

    if len(image.shape) == 2:
        channels = 1
    else:
        channels = image.shape[2]

    _x, _y = int(np.floor(x)), int(np.floor(y))

    dx = x - _x
    dy = y - _y

    if _x - 1 < 0 or _x + 2 >= image.shape[1] or _y -1 < 0 or _y + 2 >= image.shape[0]:
        return np.array([0] * channels)

    p = np.zeros(channels)

    for m in range(-1, 3):
        for n in range(-1, 3):
            p += image[_y + n, _x + m] * B_spline_cubic(m - dx) * B_spline_cubic(dy - n)

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

    if len(image.shape) == 2:
        channels = 1
    else:
        channels = image.shape[2]

    _x, _y = int(np.floor(x)), int(np.floor(y))

    dx = x - _x
    dy = y - _y

    if _x - 1 < 0 or _x + 2 >= image.shape[1] or _y -1 < 0 or _y + 2 >= image.shape[0]:
        return np.array([0] * channels)

    p = -dy * (dy-1) * (dy-2) * L(image, dx, _x, _y, 1) / 6 + \
        (dy+1) * (dy-1) * (dy-2) * L(image, dx, _x, _y, 2) / 2 + \
        -dy * (dy+1) * (dy-2) * L(image, dx, _x, _y, 3) / 2 + \
        dy * (dy+1) * (dy-1) * L(image, dx, _x, _y, 4) / 6

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

    # Rotation matrix
    R = rotation_matrix(angle)

    # Find the center of the image
    center = np.array([h // 2, w // 2])

    # Create the output image
    rotated_image = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            # Define coordinates
            coord = np.array([y, x, 1])

            # Define translation matrix
            T_to_center = translation_matrix(-center[0], -center[1])
            T_back = translation_matrix(center[0], center[1])

            # Translate to the center
            coord = np.matmul(T_to_center, coord)

            # Rotate coordinates
            coord = np.matmul(R, coord)

            # Translate back
            coord = np.matmul(T_back, coord)

            # Image coordinates
            new_y, new_x = coord[:2]

            # Interpolation
            if interpolation == 'nearest':
                rotated_image[y, x] = nearest_neighbor_interpolation(image, new_x, new_y)
            elif interpolation == 'bilinear':
                rotated_image[y, x] = bilinear_interpolation(image, new_x, new_y)
            elif interpolation == 'bicubic':
                rotated_image[y, x] = bicubic_interpolation(image, new_x, new_y)
            elif interpolation == 'lagrange':
                rotated_image[y, x] = lagrange_interpolation(image, new_x, new_y)

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
    h, w, _ = image.shape

    # Get the scaling factors
    if dimensions:
        width, height = dimensions
        scale_x = width / w
        scale_y = height / h
    else:
        scale_x = scale
        scale_y = scale

    # Scaling matrix
    S = scale_matrix(scale_x, scale_y)

    # Create the output image
    scaled_image = np.zeros((int(h * scale_y), int(w * scale_x), 3))

    for i in range(h):
        for j in range(w):
            # Define coordinates
            coord = np.array([i, j, 1])

            # Scale coordinates
            coord = np.matmul(S, coord)

            # Translate back to image coordinates
            new_i, new_j = coord[:2]

            # Interpolation
            if interpolation == 'nearest':
                scaled_image[i, j] = nearest_neighbor_interpolation(image, new_j, new_i)
            elif interpolation == 'bilinear':
                scaled_image[i, j] = bilinear_interpolation(image, new_j, new_i)
            elif interpolation == 'bicubic':
                scaled_image[i, j] = bicubic_interpolation(image, new_j, new_i)
            elif interpolation == 'lagrange':
                scaled_image[i, j] = lagrange_interpolation(image, new_j, new_i)

    return scaled_image


def transform_image(args):
    # Read the input image
    image = cv.imread(args.input, cv.IMREAD_GRAYSCALE)

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

        # Display the images
        cv.imshow('Original Image', image)
        cv.imshow('Custom Rotated Image', rotated_image)
        cv.imshow('OpenCV Rotated Image', rotated_image_cv)
        cv.imshow('Difference Image', rotated_diff_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Apply scaling
    if args.scale != 1.0:
        # Apply custom scaling
        scaled_image = scale_image(image, args.scale, None, interpolation=args.method)

        # Apply OpenCV scaling
        scaled_image_cv = cv.resize(image, None, fx=args.scale, fy=args.scale, interpolation=flags_dict[args.method])

        # Compare with OpenCV scaling (Both images and subtraction of them)
        scaled_image = scaled_image.astype(np.uint8)
        scaled_image_cv = scaled_image_cv.astype(np.uint8)
        scaled_diff_image = cv.absdiff(scaled_image, scaled_image_cv)

        # Display the images
        cv.imshow('Original Image', image)
        cv.imshow('Custom Scaled Image', scaled_image)
        cv.imshow('OpenCV Scaled Image', scaled_image_cv)
        cv.imshow('Difference Image', scaled_diff_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    # Resize to specified dimensions
    if args.dimension:
        # Apply custom resizing
        scaled_image = scale_image(image, None, args.dimension, interpolation=args.method)

        # Apply OpenCV resizing
        width, height = args.dimension
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
    
    # Save the output image
    cv.imwrite(args.output, image)

if __name__ == '__main__':
    args = parse_args()
    transform_image(args)
