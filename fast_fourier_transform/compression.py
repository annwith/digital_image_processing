import argparse
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Configure the command line arguments parser
parser = argparse.ArgumentParser(
    description='Load an image and apply a high-pass filter.'
)

# Add an argument for the image path
parser.add_argument(
    '-i', 
    '--image_path', 
    type=str,
    help='Path to the image file',
    required=True
)

# Add an argument for the compression threshold
parser.add_argument(
    '-t', 
    '--threshold', 
    type=float,
    help='Compression threshold (absolute value)',
)

# Add an argument for the compression percentage
parser.add_argument(
    '-p', 
    '--percentage', 
    type=float,
    help='Compression percentage (0-100)',
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
magnitude_spectrum = np.abs(fshift)

# Set the threshold value
if args.threshold is not None:
    threshold = args.threshold
elif args.percentage is not None:
    threshold = np.percentile(magnitude_spectrum, args.percentage)
else:
    raise ValueError("Please provide either a threshold value or a percentage value.")

# Apply the compression
compressed_fshift = (magnitude_spectrum > threshold) * fshift

# Apply the inverse Fourier transform
compressed_image = np.fft.ifft2(np.fft.ifftshift(compressed_fshift)).real

# Plotting
plt.figure(figsize=(16, 8))

# Original Image and Histogram
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.hist(image.ravel(), bins=256, color='gray')
plt.title('Original Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Magnitude Spectrum and Histogram
plt.subplot(2, 3, 2)
plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.hist(magnitude_spectrum.ravel(), bins=100, range=(0, 200000), color='gray')
plt.title('Magnitude Spectrum Histogram')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.axvline(x=threshold, color='red', linestyle='--')

# Compressed Image and Histogram
plt.subplot(2, 3, 3)
plt.imshow(compressed_image, cmap='gray')
plt.title('Compressed Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.hist(compressed_image.ravel(), bins=256, color='gray')
plt.title('Compressed Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Extrair o nome do arquivo da imagem original
filename = args.image_path.split('/')[-1]

# Salvar a imagem comprimida
cv.imwrite(f'compressed_images/{filename[:-4]}_{round(threshold)}.png', compressed_image)
