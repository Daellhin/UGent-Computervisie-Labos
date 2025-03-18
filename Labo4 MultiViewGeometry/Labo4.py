import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike


def addName(image, name="Lorin Speybrouck", size=1):
    cv2.putText(
        image,
        name,
        (10, image.shape[0] - 15),
        cv2.FONT_ITALIC,
        0.5*size,
        (0, 0, 0),
        3*size,
        cv2.LINE_AA,
    )
    image = cv2.putText(
        image,
        name,
        (10, image.shape[0] - 15),
        cv2.FONT_ITALIC,
        0.5*size,
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


# -- Assignment 1 --
# Compute SIFT keypoints and descriptors in both images and match them
def compute_keypoints_and_match(image1:MatLike, image2:MatLike, ration=0.425):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.425 * n.distance:
            good_matches.append(m)
    print (len(good_matches))
    
    return kp1, kp2, good_matches

def assignment1():
    image1 = cv2.imread("img/im00.png")
    image2 = cv2.imread("img/im01.png")

    kp1, kp2, good_matches = compute_keypoints_and_match(image1, image2)

    matched_image = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("out/assignment1.png", addName(matched_image))


# -- Assignment 2 --
# Estimate the fundamental matrix, compute the epipolar lines for the feature points you used and draw these onto both images.
def draw_epilines(image: MatLike, lines: MatLike, points: MatLike):
    r, c = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for r, pt in zip(lines, points):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        image = cv2.line(image, (x0, y0), (x1, y1), color, 1)
        image = cv2.circle(image, tuple(pt), 5, color, -1)
    return image

def assignment2():
    image1 = cv2.imread("img/im00.png")
    image2 = cv2.imread("img/im01.png")

    kp1, kp2, good_matches = compute_keypoints_and_match(image1, image2)
    points1 = np.int32([kp1[match.queryIdx].pt for match in good_matches])
    points2 = np.int32([kp2[match.trainIdx].pt for match in good_matches])
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)
    print("Fundamental matrix:\n", F)

    # Select only inlier points
    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]

    lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    img1_with_lines = draw_epilines(image1, lines1, points1)
    lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    img2_with_lines = draw_epilines(image2, lines2, points2)

    save_row([img1_with_lines, img2_with_lines], ["Image 1", "Image 2 "], "out/assignment2.png")

# -- Assignment 3 --
# Compute the essential matrix from the fundamental matrix
# Compute the translation and rotation from the essential matrix
def assignment3():
    image1 = cv2.imread("img/im00.png")
    image2 = cv2.imread("img/im01.png")

    kp1, kp2, good_matches = compute_keypoints_and_match(image1, image2)
    points1 = np.int32([kp1[match.queryIdx].pt for match in good_matches])
    points2 = np.int32([kp2[match.trainIdx].pt for match in good_matches])
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)

    K = np.array([[792, 0, 505], [0, 791, 376], [0, 0, 1]])
    E = K.T @ F @ K

    R1, R2, t = cv2.decomposeEssentialMat(E)

    print("Rotation matrix 1:\n", R1)
    print("Rotation matrix 2:\n", R2)
    print("Translation vector:\n", t)

# -- Assignment 4 --
# Convert the keypoint coordinates used in Assignment 2 to normalized coordinates
# Compute the essential matrix from the normalized coordinates
# Compute the translation and rotation from the essential matrix
def assignment4():
    image1 = cv2.imread("img/im00.png")
    image2 = cv2.imread("img/im01.png")

    kp1, kp2, good_matches = compute_keypoints_and_match(image1, image2)
    points1 = np.float32([kp1[match.queryIdx].pt for match in good_matches])
    points2 = np.float32([kp2[match.trainIdx].pt for match in good_matches])

    K = np.array([[792, 0, 505], [0, 791, 376], [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    points1_normalized = cv2.undistortPoints(np.expand_dims(points1, axis=1), K, None)
    points2_normalized = cv2.undistortPoints(np.expand_dims(points2, axis=1), K, None)
    E, mask = cv2.findFundamentalMat(points1_normalized, points2_normalized, cv2.FM_LMEDS)

    R1, R2, t = cv2.decomposeEssentialMat(E)

    print("Rotation matrix 1:\n", R1)
    print("Rotation matrix 2:\n", R2)
    print("Translation vector:\n", t)

def question3():
    R = np.array([
    [0.89905433, 0.0271129, 0.43699679],
    [0.00437026, 0.9974754, -0.0708782],
    [-0.43781526, 0.06563314, 0.89666609]
    ])

    # Calculate the angle of rotation around the Y-axis
    theta = np.arcsin(R[0, 2])
    theta_degrees = np.degrees(theta)
    print(theta_degrees)

if __name__ == '__main__':
    assignment1()
    assignment2()
    assignment3()
    assignment4()
