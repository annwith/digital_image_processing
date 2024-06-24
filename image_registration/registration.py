"""
Image registration.
"""
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """
    Function to define command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image registration.")

    parser.add_argument(
        '-i', 
        '--input', 
        nargs=2,
        required=True,
        help="Two input images in PNG format.")

    parser.add_argument(
        '-o', 
        '--output', 
        required=True,
        help="Output image in PNG format.")

    parser.add_argument(
        '-m', 
        '--method', 
        choices=['sift', 'orb'],
        default='orb',
        help="Feature detection method.")
    
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.1,
        help="Threshold percentage of matches to be used (value between 0 and 1).")

    return parser.parse_args()


def match_descriptors(descriptors1, descriptors2, method='orb', threshold=0.5):
    """
    Find matches between descriptors of two images using BFMatcher.
    
    Parameters:
        descriptors1, descriptors2: Descriptors of the two images.
        method (str): Method for feature detection.
        threshold (float): Threshold for filtering matches.
    
    Returns:
        list: List of matches.
    """
    # Matcher Brute Force with Hamming distance (or L2 for SIFT/SURF)
    if method == 'sift':
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    else:
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Find matches
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Filter matches by the threshold
    num_matches = int(len(matches) * threshold)
    matches = matches[:num_matches]

    if len(matches) < 4:
        raise ValueError("Not enough matches were found. Try lowering the threshold.")

    return matches


def find_keypoints_and_descriptors(image, method='orb'):
    """
    Find keypoints and descriptors in the image using SIFT or ORB.
    
    Parameters:
        image (numpy.ndarray): Grayscale image.
        method (str): Method for detecting keypoints.
    
    Returns:
        keypoints, descriptors: Keypoints and descriptors.
    """
    if method == 'sift':
        detector = cv.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
    elif method == 'orb':
        detector = cv.ORB_create()

    keypoints, descriptors = detector.detectAndCompute(image, None)

    return keypoints, descriptors


def ransac_and_homography(keypoints1, keypoints2, matches):
    """
    Perform RANSAC to estimate the homography matrix.
    
    Parameters:
        keypoints1, keypoints2: Keypoints in the two images.
        matches: Matches between the two images.
    
    Returns:
        numpy.ndarray: Homography matrix.
    """
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return H, mask


def warp_images(image1, image2, H):
    """Apply a perspective transformation to align the images."""
    # Dimensions of the output image
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Edge points of the images
    points1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    points2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)

    # Transform points of image 1 to the perspective of image 2
    points1_transformed = cv.perspectiveTransform(points1, H)

    # Combine points of the two images
    points_combined = np.concatenate((points2, points1_transformed), axis=0)

    # Get the bounding rectangle of the combined image
    [xmin, ymin] = np.int32(points_combined.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(points_combined.max(axis=0).ravel() + 0.5)

    # Adjust the translation matrix
    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Apply the perspective transformation
    output_image = cv.warpPerspective(image1, H_translation.dot(H), (xmax - xmin, ymax - ymin))
    output_image[translation_dist[1]:height2 + translation_dist[1], translation_dist[0]:width2 + translation_dist[0]] = image2

    return output_image


def register_images(image1_path, image2_path, output_path, method='orb', threshold=0.75):
    """
    Function to perform image registration.
    """
    # Load the two images in RGB
    img1_rgb = cv.imread(image1_path, cv.IMREAD_COLOR)
    img2_rgb = cv.imread(image2_path, cv.IMREAD_COLOR)

    # Check if the images are loaded properly
    if img1_rgb is None or img2_rgb is None:
        raise ValueError("One or both images could not be loaded. Check the paths.")

    # Convert images to grayscale for keypoint detection
    img1_gray = cv.cvtColor(img1_rgb, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2_rgb, cv.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in grayscale images
    keypoints1, descriptors1 = find_keypoints_and_descriptors(img1_gray, method)
    keypoints2, descriptors2 = find_keypoints_and_descriptors(img2_gray, method)

    # Show the keypoints
    img1_keypoints = cv.drawKeypoints(img1_gray, keypoints1, None, color=(0, 255, 0))
    img2_keypoints = cv.drawKeypoints(img2_gray, keypoints2, None, color=(0, 255, 0))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.title('Image 1 keypoints')
    plt.imshow(img1_keypoints, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Image 2 keypoints')
    plt.imshow(img2_keypoints, cmap='gray')
    plt.axis('off')

    plt.show()

    # Match descriptors
    matches = match_descriptors(descriptors1, descriptors2, method, threshold)

    # Plot both images with matches
    img_matches = cv.drawMatches(img1_gray, keypoints1, img2_gray, keypoints2, matches, None)

    plt.figure(figsize=(15, 7))
    plt.title('Matches')
    plt.imshow(img_matches)
    plt.axis('off')
    plt.show()

    # RANSAC and homography
    H, mask = ransac_and_homography(keypoints1, keypoints2, matches)

    # Warp images using the homography matrix, but in RGB
    aligned_img_rgb = warp_images(img1_rgb, img2_rgb, H)

    # Save the aligned image in RGB
    cv.imwrite(output_path, aligned_img_rgb)

    # Display the images
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 3, 1)
    plt.title('Image 1')
    plt.imshow(cv.cvtColor(img1_rgb, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Image 2')
    plt.imshow(cv.cvtColor(img2_rgb, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Aligned Image')
    plt.imshow(cv.cvtColor(aligned_img_rgb, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()


def main():
    args = parse_args()
    image1_path, image2_path = args.input
    register_images(
        image1_path,
        image2_path,
        args.output,
        args.method,
        args.threshold)


if __name__ == "__main__":
    main()
