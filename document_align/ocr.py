import cv2 as cv
import numpy as np

image = cv.imread('input_images/neg_4.png', cv.IMREAD_GRAYSCALE)

# Plot the image
cv.imshow('Image', image)