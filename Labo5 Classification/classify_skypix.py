# -*- coding: utf-8 -*-
"""
@author: David
"""

import cv2
import numpy as np
from glob import glob
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt

image_paths = sorted(glob("img/road?.png"))
label_paths = sorted(glob("img/road?_skymask.png"))

features = np.array([])
values = np.array([])

for image_path, label in zip(image_paths, label_paths):
    image = cv2.imread(image_path, 1)
    lab = cv2.imread(label, 0)

    # this makes a green/red/transparent highlight mask for visualization
    lab_color = cv2.merge((np.zeros(lab.shape, float), (lab == 255).astype(float), (lab == 0).astype(float)))

    cv2.imshow("input data", 0.7 * image / 255 + 0.3 * lab_color)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # this adds the blue, green and red color values of every pixel to the feature array
    features = np.append(features, image)
    # this adds for every pixel the mask value to the value array
    values = np.append(values, lab)

# appending flattens the array, so you have to restore the dimensions to a Kx3 array, where every pixel of the image is 1 row and the blue, green and red intensities are in the three columns
features = np.reshape(features, (values.shape[0], -1))

# this discards any pixel with value not 0 or 255 from the arrays for training
which = np.union1d(np.where(values == 255), np.where(values == 0))
features = features[which, :]
values = values[which]

# initialize and train a QDA classifier
qda = QuadraticDiscriminantAnalysis()
qda.fit(features, values)
print(f'Mean training accuracy: {qda.score(features, values)}')

for (i, image_path)  in enumerate(image_paths):
    image = cv2.imread(image_path, 1)
    # convert to a Kx3 array that has all the pixels as rows
    im_2d = np.reshape(image, (image.shape[0] * image.shape[1], -1))

    plab = qda.predict(im_2d)
    # return to original image dimensions for visualization
    plab = np.reshape(plab, (image.shape[0], image.shape[1]))
    plab_color = cv2.merge((np.zeros(plab.shape, float), (plab == 255).astype(float), (plab == 0).astype(float)))
    
    cv2.imwrite(f"out/classify-{i}.png", (0.7 * image + 0.3 * plab_color * 255).astype(np.uint8))
    # cv2.imshow("predicted data", 0.7 * image / 255 + 0.3 * plab_color)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


