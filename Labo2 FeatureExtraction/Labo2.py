import cv2
import numpy as np
import matplotlib.pyplot as plt


def addName(image, name="Lorin Speybrouck"):
    cv2.putText(
        image,
        name,
        (10, image.shape[0] - 15),
        cv2.FONT_ITALIC,
        0.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    image = cv2.putText(
        image,
        name,
        (10, image.shape[0] - 15),
        cv2.FONT_ITALIC,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def save_row(images: list, names: list, filename="out/temp.png"):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
    for i, (image, name) in enumerate(zip(images, names)):
        axes[i].imshow(image, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(name)
    plt.tight_layout(pad=2)
    plt.savefig(filename)
    cv2.imwrite(filename, addName(cv2.imread(filename)))


def plot_row(images: list):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for i, image in enumerate(images):
        axes[i].imshow(image, cmap="gray")
        axes[i].axis("off")
    plt.show()


def showFilter(name, f):
    cv2.imshow(name, 0.5 * f / f.max() + 0.5)
    cv2.imwrite(name + " . png ", 255 * (0.5 * f / f.max() + 0.5))


def rgb_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# -- assignment 1 --
# Calculate the vertical first-order derivative using sobel operator
building = cv2.imread("img/building.png", cv2.IMREAD_GRAYSCALE)
sobel_vertical = cv2.Sobel(building, cv2.CV_64F, 0, 1, ksize=3)
abs_sobel_vertical = cv2.convertScaleAbs(sobel_vertical)
cv2.imwrite("out/assignment1.png", addName(abs_sobel_vertical))


# -- assignment 2 --
# Create a 15x15 DoG filter
def create_dog(ksize: int, sigma_1: float, sigma_2: float, angle: float, filename: str):
    gaussian_1D_1 = cv2.getGaussianKernel(ksize, sigma_1)
    matrix = np.zeros((15, 15))
    matrix[7] = gaussian_1D_1.T

    gaussian_1D_2 = cv2.getGaussianKernel(ksize, sigma_2)
    elliptical_2D_Gaussian = cv2.filter2D(matrix, -1, gaussian_1D_2)

    sobel_vertical = cv2.Sobel(elliptical_2D_Gaussian, cv2.CV_64F, 0, 1, ksize=3)

    rotation_matrix = cv2.getRotationMatrix2D((7, 7), angle, 1)
    DOG = cv2.warpAffine(
        sobel_vertical, rotation_matrix, (matrix.shape[1], matrix.shape[0])
    )

    save_row(
        [matrix, elliptical_2D_Gaussian, sobel_vertical, DOG],
        ["1D Gaussian", "Elliptical Gaussian", "Sobel filtered", "Rotated"],
        filename,
    )
    return DOG


create_dog(15, 3, 2, 45, "out/assignment2.png")

# -- assignment 3 --
# Filter rays.png with a well chosen DoG ﬁlter
DOG = create_dog(15, 5.5, 0.4, 75, "out/temp.png")
strips = cv2.imread("img/rays.png", cv2.IMREAD_GRAYSCALE)
filtered_strips = np.absolute(cv2.filter2D(strips, cv2.CV_64F, DOG))
save_row([strips, filtered_strips], ["Strips", "Edges"], "out/assignment3.png")

# -- assignment 4 --
# Apply Canny edge detection
rays = cv2.imread("img/rays.png", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(rays, 20, 100)
save_row([rays, edges], ["Rays", "Edges"], "out/assignment4.png")

# -- assignment 5 --
# Apply Canny edge detection
painting_gray = cv2.imread("img/painting4.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(painting_gray, 400, 500)
save_row([painting_gray, edges], ["Painting", "Edges"], "out/assignment5.png")

# -- assignment 6 --
# Apply HoughLines to the result of Assignment 5
painting_color = cv2.imread("img/painting4.jpg")
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(painting_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("out/assignment6.png", addName(painting_color))

# plot_row([edges, rgb_image(painting_color)])


# -- assignment 7 --
# Detect Harris corners in shot1.png and shot2.png
def detect_harris_corners(image, blocksize, ksize, k, threshold=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blocksize, ksize, k)
    dst = cv2.dilate(dst, None)
    image[dst > threshold * dst.max()] = [0, 0, 255]
    return image


shot1 = cv2.imread("img/shot1.png")
shot2 = cv2.imread("img/shot2.png")
shot1_corners = detect_harris_corners(shot1, 2, 3, 0.04, 0.04)
shot2_corners = detect_harris_corners(shot2, 2, 3, 0.04, 0.04)

save_row(
    [rgb_image(shot1_corners), rgb_image(shot2_corners)],
    ["Shot 1", "Shot 2"],
    "out/assignment7.png",
)

# -- assignment 8 --
# Detect ORB features in each of the two original images
shot1 = cv2.imread("img/shot1.png")
shot2 = cv2.imread("img/shot2.png")

orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(shot1, None)
keypoints2, descriptors2 = orb.detectAndCompute(shot2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
match_img = cv2.drawMatches(
    shot1,
    keypoints1,
    shot2,
    keypoints2,
    matches[:32],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv2.imwrite("out/assignment8.png", addName(match_img))
