import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from glob import glob
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def addName(image, name="Valerie Decloedt Van Laere, Lorin Speybrouck", size=1):
    cv2.putText(
        image,
        name,
        (10, image.shape[0] - 15),
        cv2.FONT_ITALIC,
        0.5 * size,
        (0, 0, 0),
        3 * size,
        cv2.LINE_AA,
    )
    image = cv2.putText(
        image,
        name,
        (10, image.shape[0] - 15),
        cv2.FONT_ITALIC,
        0.5 * size,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def save_rows(images: list, filename="out/temp.png", names: list = None, num_cols=5):
    if names is None:
        names = [""] * len(images)
    num_rows = (len(images) + num_cols - 1) // num_cols
    axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))[1].flatten()
    for i, (image, name) in enumerate(zip(images, names)):
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(name)
    [ax.axis("off") for ax in axes]
    plt.tight_layout(pad=1)
    plt.savefig(filename)
    cv2.imwrite(filename, addName(cv2.imread(filename)))


def plot_row(images: list):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for i, image in enumerate(images):
        axes[i].imshow(image, cmap="gray")
        axes[i].axis("off")
    plt.show()


# -- Assignment 1 --
# Make a filterbank of DoG filters in 2 scales and 6 orientation
def create_dog_filter(ksize: int, sigma_1: float, sigma_2: float, angle: float):
    half_ksize = int(ksize / 2)
    gaussian_1D_1 = cv2.getGaussianKernel(ksize, sigma_1)
    matrix = np.zeros((ksize, ksize))
    matrix[half_ksize] = gaussian_1D_1.T

    gaussian_1D_2 = cv2.getGaussianKernel(ksize, sigma_2)
    elliptical_2D_Gaussian = cv2.filter2D(matrix, -1, gaussian_1D_2)

    sobel_vertical = cv2.Sobel(elliptical_2D_Gaussian, cv2.CV_64F, 0, 1, ksize=3)

    rotation_matrix = cv2.getRotationMatrix2D((half_ksize, half_ksize), angle, 1)
    DOG = cv2.warpAffine(sobel_vertical, rotation_matrix, (matrix.shape[1], matrix.shape[0]))
    return DOG


def assignment1():
    DOGs = [create_dog_filter(19, 3.5, 1.5, i * 30) for i in range(6)] + [
        create_dog_filter(9, 2, 0.8, i * 30) for i in range(6)
    ]
    save_rows(DOGs, "out/assignment1.png")


# -- Assignment 2 --
def create_features_and_values(
    image_paths: list[str], label_paths: list[str], DOGs: list[MatLike], assignment: str
):
    test_images = [cv2.imread(path) for path in image_paths[:2]]
    test_images_grayscale = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_images]
    labels = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in label_paths]

    # Apply DOG filter
    images_responses = [
        [cv2.filter2D(image, cv2.CV_64F, dog) for dog in DOGs] for image in test_images_grayscale
    ]

    # Save filter responses
    for i, image_response in enumerate(images_responses):
        for j, response in enumerate(image_response):
            plt.imsave(f"debug/{assignment}-{i}-{j}.png", response, cmap="gray")

    # Create features and values
    RGB_features = [RGB for image in test_images for RGB in image.reshape(-1, 3)]
    response_features = [
        responses
        for image_responses in images_responses
        for responses in np.stack(image_responses, axis=-1).reshape(-1, len(DOGs))
    ]
    features = np.concatenate((RGB_features, response_features), axis=1)
    # features = np.array(RGB_features)
    values = np.concatenate(labels).flatten()

    # Filter
    which = np.union1d(np.where(values == 255), np.where(values == 0))
    features = features[which, :]
    values = values[which]

    return features, values


def test_classifier(
    classifier: BaseEstimator, image_paths: list[str], DOGs: list[MatLike], assignment: str
):
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_response = [cv2.filter2D(image_grayscale, cv2.CV_64F, dog) for dog in DOGs]

        RGB_features = image.reshape(-1, 3)
        response_features = np.stack(image_response, axis=-1).reshape(-1, len(DOGs))
        features = np.concatenate((RGB_features, response_features), axis=1)
        # features = RGB_features

        predictions = classifier.predict(features).reshape((image.shape[0], image.shape[1]))
        predictions_mask = cv2.merge(
            (
                np.zeros(predictions.shape, float),
                (predictions == 255).astype(float),
                (predictions == 0).astype(float),
            )
        )

        cv2.imwrite(
            f"out/{assignment}-{i}.png",
            addName((0.7 * image + 0.3 * predictions_mask * 255).astype(np.uint8)),
        )


def assignment2():
    # Setup
    image_paths = sorted(glob("img/road?.png"))
    label_paths = sorted(glob("img/road?_skymask.png"))
    DOGs = [create_dog_filter(19, 3.5, 1.5, i * 30) for i in range(6)] + [
        create_dog_filter(9, 2, 0.8, i * 30) for i in range(6)
    ]
    features, values = create_features_and_values(image_paths, label_paths, DOGs, "assignment2")

    # Train
    classifier = QuadraticDiscriminantAnalysis()
    classifier.fit(features, values)
    print(f"Mean training accuracy: {classifier.score(features, values)}")

    # Test
    test_classifier(classifier, image_paths, DOGs, "assignment2")


# -- Assignment 3 --
def assignment3():
    # Setup
    image_paths = sorted(glob("img/road?.png"))
    label_paths = sorted(glob("img/road?_skymask.png"))
    DOGs = [create_dog_filter(19, 3.5, 1.5, i * 30) for i in range(6)] + [
        create_dog_filter(9, 2, 0.8, i * 30) for i in range(6)
    ]
    features, values = create_features_and_values(image_paths, label_paths, DOGs, "assignment3")

    # Train
    classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=0.01, random_state=42)
    classifier.fit(features, values)
    print(f"Mean training accuracy: {classifier.score(features, values)}")

    # Test
    test_classifier(classifier, image_paths, DOGs, "assignment3")


# -- Assignment 4 --
def assignment4():
    # Setup
    image_paths = sorted(glob("img/road?.png"))
    label_paths = sorted(glob("img/road?_skymask.png"))
    DOGs = [create_dog_filter(19, 3.5, 1.5, i * 30) for i in range(6)] + [
        create_dog_filter(9, 2, 0.8, i * 30) for i in range(6)
    ]
    features, values = create_features_and_values(image_paths, label_paths, DOGs, "assignment4")

    # Train
    classifier = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=100,
        random_state=42,
        tol=0.0001,
        verbose=True,
    )
    classifier.fit(features, values)
    print(f"Mean training accuracy: {classifier.score(features, values)}")

    # Test
    test_classifier(classifier, image_paths, DOGs, "assignment4")

    plt.plot(classifier.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig("out/assignment4-loss.png")
    cv2.imwrite("out/assignment4-loss.png", addName(cv2.imread("out/assignment4-loss.png")))


if __name__ == "__main__":
    # print("Assignment 1:")
    # assignment1()
    # print("Assignment 2:")
    # assignment2()
    # print("Assignment 3:")
    # assignment3()
    print("Assignment 4:")
    assignment4()
