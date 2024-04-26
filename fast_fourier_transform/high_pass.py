import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from utils import create_mask


# Callback function for the slider
def update_radius(radius = 0):
    """
    Update the high-pass filter and the high-pass filtered image.

    Args:
        radius: Radius of the high-pass filter.
    """

    # Update the high-pass mask
    high_pass_mask = create_mask(
        height=image.shape[0],
        width=image.shape[1],
        radius=radius,
        high_pass=True)

    # Apply the high-pass filter to the magnitude spectrum
    high_pass_filter = magnitude_spectrum.copy()
    high_pass_filter[~high_pass_mask] = 0  # Set False values to 0

    # Apply the high-pass filter to the image
    image_high_pass = fshift * high_pass_mask

    # Apply the inverse Fourier transform
    image_high_pass = np.fft.ifft2(np.fft.ifftshift(image_high_pass)).real
    
    print("Image data type:", image.dtype)
    print("High pass image data type:", image_high_pass.dtype)
    print("High pass image max and min:", image_high_pass.max(), image_high_pass.min())

    print(image)
    print(image_high_pass)

    # Displays the original image
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Displays the high-pass filtered image
    axs[0, 1].imshow(image_high_pass, cmap='gray')
    axs[0, 1].set_title('High-pass Filtered Image')
    axs[0, 1].axis('off')

    # Displays the magnitude spectrum
    axs[1, 0].imshow(magnitude_spectrum, cmap='gray')
    axs[1, 0].set_title('Magnitude Spectrum')
    axs[1, 0].axis('off')

    # Displays the filter
    axs[1, 1].imshow(high_pass_filter, cmap='gray')
    axs[1, 1].set_title('High-pass Filter')
    axs[1, 1].axis('off')

    # Update the plot
    plt.draw()


# Configure the command line arguments parser
parser = argparse.ArgumentParser(
    description='Load a image and apply a high-pass filter.')

# Add an argument for the image path
parser.add_argument(
    '-i', 
    '--image_path', 
    type=str,
    help='Path to the image file',
    required=True
)

# Command line arguments parser
args = parser.parse_args()

# Loads image
image = cv.imread(args.image_path, cv.IMREAD_GRAYSCALE)

# Assert that the image was read
assert image is not None, "File could not be read."

# Apply FFT to the input image
f = np.fft.fft2(image)

# Centralize the frequency spectrum
fshift = np.fft.fftshift(f)

# Calculate the magnitude of the frequency spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Create a figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Create a slider for adjusting the radius
slider_ax = fig.add_axes([0.1, 0.01, 0.8, 0.03])  # [left, bottom, width, height]
radius_slider = Slider(
    slider_ax,
    'Radius',
    0,
    image.shape[0] // 2,
    valinit=30,
    valstep=1)
radius_slider.on_changed(update_radius)

# Display the images
update_radius(radius_slider.val)

# Show the plot
plt.show()