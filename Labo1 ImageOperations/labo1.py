import cv2
from utils import *
import numpy as np

def addName(image, name="Lorin Speybrouck"):
    cv2.putText(image, name, (10, image.shape[0]-15), cv2.FONT_ITALIC, 0.5, (0,0,0), 3, cv2.LINE_AA)
    image = cv2.putText(image, name, (10, image.shape[0]-15), cv2.FONT_ITALIC, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return image

# -- assignment 1 --
# Crop the image so it becomes square by chopping off the a part on the right side.
clouds_image = cv2.imread('img/clouds.jpg')

height, width, _ = clouds_image.shape
cropped_image = clouds_image[0:height, 0:height]

cv2.imwrite('out/assignment1.jpg', addName(cropped_image))

# -- assignment 2 --
# Discolor the image by reducing the intensity of the red value of every pixel by half
discolored_image = clouds_image.copy()
discolored_image[:, :, 2] = discolored_image[:, :, 2] // 2

cv2.imwrite('out/assignment2.jpg', addName(discolored_image))

# -- Assignment 3 --
# Discolor the image by doubling the intensity of the red value of every pixel
doubled_red_image = clouds_image.copy()
doubled_red_image[:, :, 2] = cv2.min(doubled_red_image[:, :, 2] * 2, 255)  # Clamp at 255

cv2.imwrite('out/assignment3.jpg', addName(doubled_red_image))

# -- Assignment 4 --
# Make a regular grid of black dots on the image so that the dots are 10 pixels apart vertically and horizontally
grid_image = clouds_image.copy()
dot_spacing = 10

for y in range(10, height, dot_spacing):
    for x in range(10, width-10, dot_spacing):
        cv2.circle(grid_image, (x, y), 1, (0, 0, 0), -1)

cv2.imwrite('out/assignment4.jpg', addName(grid_image))

# -- Assignment 5 --
# Convert the image to a grayscale image
grayscale_image = cv2.cvtColor(clouds_image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('out/assignment5.jpg', addName(grayscale_image))

# -- Assignment 6 --
## Threshold the grayscale image at 50% of the maximum value for this datatype.
_, thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite('out/assignment6.jpg', addName(thresholded_image))

# -- Assignment 7 --
# Threshold the grayscale image at the ideal threshold determined by Otsu’s method
_, otsu_thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite('out/assignment7.jpg', addName(otsu_thresholded_image))

# -- Assignment 8 --
# Adaptively threshold the grayscale version of painting2.jpg
painting_image = cv2.imread('img/painting2.jpg', cv2.IMREAD_GRAYSCALE)

adaptive_thresholded_image = cv2.adaptiveThreshold(
    painting_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
)

cv2.imwrite('out/assignment8.jpg', addName(adaptive_thresholded_image))

# -- Assignment 9 --
# Remove the white noise from whitenoise.png by Gaussian filtering
whitenoise_image = cv2.imread('img/whitenoise.png')

kernel_size = (21, 21)
sigma = 4
gaussian_filtered_image = cv2.GaussianBlur(whitenoise_image, kernel_size, sigma)

cv2.imwrite('out/assignment9.jpg', addName(gaussian_filtered_image))

# def gaussian_filter(kernel_size, sigma):
# 	whitenoise_image = cv2.imread('img/whitenoise.png', cv2.IMREAD_GRAYSCALE)
# 	gaussian_filtered_image = cv2.GaussianBlur(whitenoise_image, (kernel_size, kernel_size), sigma)

# 	return [whitenoise_image, gaussian_filtered_image]

# show_results_interactive(gaussian_filter, [5, 1.5], "Gaussian Filter")

# -- Assignment 10 --
## Test the Gaussian filter on saltandpeppernoise.png.
saltpeppernoise_image = cv2.imread('img/saltandpeppernoise.png')

kernel_size = (21, 21)
sigma = 4
gaussian_filtered_image = cv2.GaussianBlur(saltpeppernoise_image, kernel_size, sigma)
cv2.imwrite('out/assignment10.jpg', addName(gaussian_filtered_image))

# -- Assignment 11 --
## Apply median filtering on the same image.
saltpeppernoise_image = cv2.imread('img/saltandpeppernoise.png')
median_filtered_image = cv2.medianBlur(saltpeppernoise_image, 3)
cv2.imwrite('out/assignment11.jpg', addName(median_filtered_image))


# -- Assignment 12 --
## Implement unsharp masking to sharpen unsharp.png.
unsharp_image = cv2.imread('img/unsharp.png', cv2.IMREAD_COLOR)

# Blur the image
unsharp_image_float = unsharp_image.astype(np.float32)
blurred_image = cv2.GaussianBlur(unsharp_image_float, (21, 21), 10)

# Subtract the blurred from the original
difference_image = unsharp_image_float - blurred_image

# Amplify the difference by multiplying it with a factor
amplified_difference = difference_image * 1.5

# Add this amplified difference image to the original image
sharpened_image = unsharp_image_float + amplified_difference

sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
cv2.imwrite('out/assignment12.jpg', addName(sharpened_image))

# -- Assignment 13 --
## Write a program that blurs blots.png diagonally with the kernel below (mind the multiplication factor in front).
image = cv2.imread("img/blots.png")

kernel = (1/7) * np.eye(7, dtype=np.float32)
blurred_image = cv2.filter2D(image, -1, kernel)

cv2.imwrite("out/assignment13.png", addName(blurred_image))