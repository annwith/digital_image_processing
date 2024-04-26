import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from utils import create_mask

# Callback function for the slider
def update_radius(smaller_radius = 10, bigger_radius = 20):
    """
    Update the band-stop filter and the band-stop filtered image.

    Args:
        radius: Radius of the band-stop filter.
    """

    # Create the band-stop mask
    low_pass_mask = create_mask(
        height=image.shape[0],
        width=image.shape[1],
        radius=smaller_radius,
        high_pass=False)
    high_pass_mask = create_mask(
        height=image.shape[0],
        width=image.shape[1],
        radius=bigger_radius,
        high_pass=True)
    band_stop_mask = low_pass_mask | high_pass_mask

    # Apply the band-stop filter to the magnitude spectrum
    band_stop_filter = magnitude_spectrum.copy()
    band_stop_filter[~band_stop_mask] = 0 # Set False values to 0

    # Apply the band-stop filter to the image
    image_band_stop = fshift * band_stop_mask

    # Apply the inverse Fourier transform
    image_band_stop = np.fft.ifft2(np.fft.ifftshift(image_band_stop)).real

    # Displays the original image
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Displays the band-stop filtered image
    axs[0, 1].imshow(image_band_stop, cmap='gray')
    axs[0, 1].set_title('Band-stop Filtered Image')
    axs[0, 1].axis('off')

    # Displays the magnitude spectrum
    axs[1, 0].imshow(magnitude_spectrum, cmap='gray')
    axs[1, 0].set_title('Magnitude Spectrum')
    axs[1, 0].axis('off')

    # Displays the filter
    axs[1, 1].imshow(band_stop_filter, cmap='gray')
    axs[1, 1].set_title('Band-stop Filter')
    axs[1, 1].axis('off')

    plt.draw()


# Configure the command line arguments parser
parser = argparse.ArgumentParser(
    description='Load a image and apply a band-stop filter.')

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
fshift = np.fft.fftshift(f)

# Calculate the magnitude of the frequency spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Create a figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Create sliders for adjusting the smaller and bigger radius
slider_ax_smaller = fig.add_axes([0.1, 0.06, 0.8, 0.03])
radius_slider_smaller = Slider(
    slider_ax_smaller,
    'Smaller Radius',
    0,
    image.shape[0] // 2,
    valinit=10,
    valstep=1)

slider_ax_bigger = fig.add_axes([0.1, 0.01, 0.8, 0.03])
radius_slider_bigger = Slider(
    slider_ax_bigger,
    'Bigger Radius',
    0,
    image.shape[0] // 2,
    valinit=20,
    valstep=1)

# Connect sliders to the update function
radius_slider_smaller.on_changed(lambda x: update_radius(radius_slider_smaller.val, radius_slider_bigger.val))
radius_slider_bigger.on_changed(lambda x: update_radius(radius_slider_smaller.val, radius_slider_bigger.val))

# Display the images
update_radius(radius_slider_smaller.val, radius_slider_bigger.val)

# Show the plot
plt.show()