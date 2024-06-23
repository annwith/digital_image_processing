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
        choices=['sift', 'surf', 'brief', 'orb'],
        default='orb',
        help="Feature detection method.")

    return parser.parse_args()


def find_keypoints(image, method='orb'):
    if method == 'sift':
        detector = cv.SIFT_create()
    elif method == 'surf':
        detector = cv.SURF_create()
    elif method == 'brief':
        detector = cv.xfeatures2d.BriefDescriptorExtractor_create()
    elif method == 'orb':
        detector = cv.ORB_create()

    keypoints, descriptors = detector.detectAndCompute(image, None)

    return keypoints, descriptors


def register_images(image1_path, image2_path, output_path, method='orb'):
    """
    Function to perform image registration.
    """
    # Load the two images
    img1 = cv.imread(image1_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(image2_path, cv.IMREAD_GRAYSCALE)

    # Check if the images are loaded properly
    if img1 is None or img2 is None:
        raise ValueError("One or both images could not be loaded. Check the paths.")

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = find_keypoints(img1, method)
    keypoints2, descriptors2 = find_keypoints(img2, method)

    # Show the keypoints
    img1_keypoints = cv.drawKeypoints(img1, keypoints1, None, color=(0,255,0))
    img2_keypoints = cv.drawKeypoints(img2, keypoints2, None, color=(0,255,0))

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

    # Matcher Brute Force with Hamming distance (or L2 for SIFT/SURF)
    if method in ['sift', 'surf']:
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    else:
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Plot both images with matches
    img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)

    plt.figure(figsize=(15, 7))
    plt.title('Matches')
    plt.imshow(img_matches)
    plt.axis('off')
    plt.show()

    # Extract the matched points
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Estimate the transformation matrix
    h, _ = cv.findHomography(points2, points1, cv.RANSAC)

    # Align the second image to the first image
    height, width = img1.shape
    aligned_img = cv.warpPerspective(
        img2, h, (width, height), flags=cv.INTER_CUBIC)

    # Save the aligned image
    cv.imwrite(output_path, aligned_img)

    # Display the images
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 3, 1)
    plt.title('Image 1')
    plt.imshow(img1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Image 2')
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Aligned Image')
    plt.imshow(aligned_img, cmap='gray')
    plt.axis('off')

    plt.show()

    # Draw lines between matched points
    img_matches_lines = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 7))
    plt.title('Matches with lines')
    plt.imshow(img_matches_lines)
    plt.axis('off')
    plt.show()


def main():
    args = parse_args()
    image1_path, image2_path = args.input
    register_images(
        image1_path,
        image2_path,
        args.output,
        args.method)


if __name__ == "__main__":
    main()
