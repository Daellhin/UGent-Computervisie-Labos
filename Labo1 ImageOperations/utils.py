from typing import Callable
import cv2
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def cumulative_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cumhist = np.cumsum(hist).astype(np.float64)
    cumhist = cumhist / cumhist[-1]
    return cumhist


def show_results_interactive(function: Callable, values: list[int], title="Image"):
    image, result_image = function(*values)
    # Setup figures
    fig = plt.figure(figsize=(20, 15))
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(321)
    ax.set_title("Original Image")
    ax0 = fig.add_subplot(322)
    ax0.set_title("Image after Transformation")
    ax1 = fig.add_subplot(323)
    ax1.set_title("Histogram of Original")
    ax2 = fig.add_subplot(324, sharex=ax1)
    ax2.set_title("Histogram after Transformation")
    ax3 = fig.add_subplot(325, sharex=ax1)
    ax3.set_title("Cumulative Histogram of Original")
    ax4 = fig.add_subplot(326, sharex=ax1)
    ax4.set_title("Cumulative Histogram after Transformation")
    ax1.set_xlim([0, 255])

    # Plot data
    def plot_data(image, result_image):
        [x.clear() for x in [ax1, ax2, ax3, ax4]]
        ax1.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
        ax2.plot(cv2.calcHist([result_image], [0], None, [256], [0, 256]))
        ax3.plot(cumulative_histogram(image))
        ax4.plot(cumulative_histogram(result_image))
        ax.imshow(image, cmap=plt.get_cmap("gray"))
        ax0.imshow(result_image, cmap=plt.get_cmap("gray"))

    plot_data(image, result_image)
    # Add sliders
    sliders = [0] * len(values)
    for i, value in enumerate(values):
        fig.subplots_adjust(bottom=0.25)
        slider_axis = fig.add_axes([0.2, 0.2 - (i / 20), 0.65, 0.03])
        sliders[i] = Slider(
            slider_axis,
            "Slider" + str(i),
            valmin=1,
            valmax=21,
            valinit=value,
            valstep=2,
        )

        def sliders_on_changed(val, i):
            values[i] = val
            # print(*values)
            image, result_image = function(*values)
            plot_data(image, result_image)
            fig.canvas.draw_idle()

        sliders[i].on_changed(lambda val, idx=i: sliders_on_changed(val, idx))
    plt.show()
