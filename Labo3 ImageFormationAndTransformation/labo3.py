import math as m
import random as r
from multiprocessing import Pool

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt


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
# Create a 3D plot of a cube and project it to a 2D image
def assignment1():
    vertices = np.array(
        [
            [-0.5, -0.5, 2.5],  # vertex 0 (front bottom left)
            [0.5, -0.5, 2.5],  # vertex 1 (front bottom right)
            [0.5, 0.5, 2.5],  # vertex 2 (front top right)
            [-0.5, 0.5, 2.5],  # vertex 3 (front top left)
            [-0.5, -0.5, 3.5],  # vertex 4 (back bottom left)
            [0.5, -0.5, 3.5],  # vertex 5 (back bottom right)
            [0.5, 0.5, 3.5],  # vertex 6 (back top right)
            [-0.5, 0.5, 3.5],  # vertex 7 (back top left)
        ]
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # front face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # back face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # connecting edges
        ]
    )
    FOV = 90
    f = (1920 / 2) / m.tan(m.radians(FOV / 2))
    K = np.array([[f, 0, 960], [0, f, 540], [0, 0, 1]])

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*vertices.T, marker="o", color="k", ls="")
    [ax.plot(*vertices[[start, end], :].T, color="r") for start, end in edges]
    filename = "out/assignement1_3d.png"
    plt.savefig(filename)
    cv2.imwrite(filename, addName(cv2.imread(filename)))

    # Project
    verteces_p = []
    for vertex in vertices:
        vertex_p = K @ vertex
        vertex_p = vertex_p / vertex_p[2]
        verteces_p.append(vertex_p)

    # 2D plot
    image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

    for start, end in edges:
        start_point = (int(verteces_p[start][0]), int(verteces_p[start][1]))
        end_point = (int(verteces_p[end][0]), int(verteces_p[end][1]))
        cv2.line(image, start_point, end_point, (0, 0, 255), 2)
    for point in verteces_p:
        center = (int(point[0]), int(point[1]))
        cv2.circle(image, center, 5, (0, 0, 0), -1)

    cv2.imwrite("out/assignement1_2d.png", addName(image))



# -- Assignment 2 --
# Calibrate a camera using checkerboard images, and determine the intrinsic parameters
def calculate_camera_calibration(images: list[str], patternSize: tuple[int, int]):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    w, h = patternSize
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, patternSize, None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Debug show chessboard
            # cv.drawChessboardCorners(img, patternSize, corners2, ret)
            # cv.imshow(fname, img)
            # cv.waitKey(500)

    # print(f"Found {len(objpoints)} of {len(images)}")
    cv.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    total_error = mean_error/len(objpoints)

    return mtx, dist, total_error

def assignment2():
    patternSize = (10, 6)
    image_numbers = list(range(3, 60))
    images = [f"calibration_frames/img_{num:04d}.png" for num in image_numbers]

    mtx, dist, error = calculate_camera_calibration(images, patternSize)
    print(f"Camera matrix(Error={error}):")
    print(mtx)
    print("Distortion coefficients:")
    print(dist)

# -- Assignment 3 --
# Calibrate 5 times with random subset of images
def calculate_random_camera_calibration(i):
    patternSize = (10, 6)
    random_numbers = [r.randint(3, 59) for _ in range(20)]
    images = [f"calibration_frames/img_{num:04d}.png" for num in random_numbers]

    mtx, dist, error = calculate_camera_calibration(images, patternSize)
    # print(i)
    return mtx, dist, error

def assignment3():
    for i in range(5):
        mtx, dist, error = calculate_random_camera_calibration()
        print(f"Camera matrix {i}(Error={error}):")
        print(mtx)

# -- Assignment 4 --
# Using the best calibration, undistort an image
def find_best_calibration(iterations=5):
    with Pool(processes=12) as P:
        calibrations = P.map(calculate_random_camera_calibration, range(iterations))
        best_calibration = min(calibrations, key=lambda x: x[2])
        return best_calibration

def assignment4():
    mtx, dist, error = find_best_calibration(1000)
    print(f"Best camera matrix(Error={error}):")
    print(mtx)
    print("Distortion coefficients:")
    print(dist)

    img = cv.imread('calibration_frames/img_0053.png')
    h,  w = img.shape[:2]
    w1,h1 = 3*w,3*h
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w1,h1), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    
    cv.imwrite('assignment4_pre.png', dst)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv.imwrite('assignment4.png', dst)

# -- Assignment 5 --
# Shear the image so that the photographer’s shadow becomes vertical
def assignment5():
    image = cv2.imread('img/shadow.png')
    h, w = image.shape[:2]

    m = -0.25 # Shear factor
    x_translation = abs(m * h) if m < 0 else 0
    M = np.float32([[1, m, x_translation], [0, 1, 0]])
    new_width = int(w + abs(m * h))
    
    result = cv2.warpAffine(image, M, (new_width, h))
    cv2.imwrite('out/assignment5.png', addName(result))

# -- Assignment 6 --
# Transform the image so that the shadow box is seen from above
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((x, y))
        print(f"Point selected: {(x, y)}")
        if len(param) == 4:
            cv2.destroyAllWindows()

def assignment6():
    image = cv2.imread('img/shadow_box.png')
    selected_points = []

    cv2.namedWindow('Select 4 points')
    cv2.setMouseCallback('Select 4 points', onMouse, selected_points)

    while len(selected_points) < 4:
        cv2.imshow(f'Select 4 points', image)
        cv2.waitKey(1)

    x_offset = 100
    y_offset = 100

    src_points = np.array(selected_points, dtype='float32')
    width = max(np.linalg.norm(src_points[0] - src_points[1]), np.linalg.norm(src_points[2] - src_points[3])) + 2 * x_offset
    height = max(np.linalg.norm(src_points[0] - src_points[3]), np.linalg.norm(src_points[1] - src_points[2])) + 2 * y_offset

    dst_points = np.array([
        [x_offset, y_offset],
        [width - x_offset, y_offset],
        [width -x_offset, height  - y_offset],
        [x_offset, height - y_offset]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))

    cv2.imwrite('out/assignment6.png', addName(warped))

if __name__ == '__main__':
    assignment1()
    assignment2()
    assignment3()
    assignment4()
    assignment5()
    assignment6()
