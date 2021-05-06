import math
import numpy as np
import random
from typing import Union

# from PIL import Image
from matplotlib.colors import CSS4_COLORS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .preprocess import Image




def cut_threshold(detection: dict, threshold: float = math.inf, count: int = math.inf):
    filtered_detection = dict()
    score_list = [
        i for i, d in enumerate(detection["detection_scores"]) if d >= threshold
    ]
    if score_list:
        max_score = max(score_list)
    else:
        max_score = -1

    n_threshold = min(count, max_score)
    for key in detection.keys():
        if not hasattr(detection[key], "__getitem__"):
            filtered_detection[key] = detection[key]
        else:
            filtered_detection[key] = detection[key][: n_threshold + 1]

    return filtered_detection


def plot_object_detection(img: Image, predict: np.ndarray, **kwargs):
    color = list((CSS4_COLORS.keys()))
    random.shuffle(color)
    img_plot = img.pil_image
    width = img_plot.width
    height = img_plot.height
    fig, ax = plt.subplots(**kwargs)
    ax.imshow(img_plot)

    patch_list = list()
    for box, cls, color in zip(
        predict["detection_boxes"], predict["detection_classes"], color
    ):
        y_min, x_min, y_max, x_max = box
        x_delta = width * (x_max - x_min)
        y_delta = height * (y_max - y_min)
        patch = patches.Rectangle(
            (width * x_min, height * y_min),
            x_delta,
            y_delta,
            fill=False,
            label=cls["name"],
            linewidth=1,
            color=color,
        )
        patch_list.append(patch)
        ax.add_patch(patch)
    ax.legend()

    plt.show()
