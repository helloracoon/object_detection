import time

import cv2
import matplotlib
import pafy
from matplotlib.colors import to_rgb
import numpy as np

from object_detection.functions import (
    plot_object_detection,
    cut_threshold,
    filter_predict,
    get_object_patch,
    RANDOM_COLORS,
)
from object_detection.detector import TfDefaultDetector
from object_detection.preprocess import Image


def test_image():
    # detector = TfDefaultDetector("resnet50v2_512x512/1", "mscoco_label_map.pbtxt")
    detector = TfDefaultDetector("centernet/resnet101v1_fpn_512x512/1", "mscoco_label_map.pbtxt")
    detector.initialize()
    image = Image(
        "https://imgnews.pstatic.net/image/421/2021/05/06/0005334029_001_20210506113113527.jpg?type=w647",
        stream=True,
    )
    predict = detector.predict(
        image,
        use_name=True,
    )
    predict_cut = cut_threshold(predict, 0.6)
    ax = plot_object_detection(image, predict_cut, dpi=700)


def test_video():
    # detector = TfDefaultDetector("ssd_mobilenet_v2/2", "mscoco_label_map.pbtxt")
    detector = TfDefaultDetector("ssd_mobilenet_v2/fpnlite_320x320/1", "mscoco_label_map.pbtxt")
    detector.initialize()

    # CV2 Setup
    cam = cv2.VideoCapture(0)
    fps = 15
    cam.set(fps, 0)
    while True:
        ret_val, img = cam.read()
        if ret_val is False:
            continue

        image = Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        predict = detector.predict(image, use_name=True)
        color_map = {
            item["name"]: color
            for item, color in zip(detector.category_map.values(), RANDOM_COLORS)
        }
        # predict_person = filter_predict(predict, 'person')
        predict_cut = cut_threshold(predict, 0.)
        rectangles = get_object_patch(image, predict_cut)

        for rectangle in rectangles:
            x, y = list(int(xy) for xy in rectangle.get_xy())
            width = int(rectangle.get_width())
            height = int(rectangle.get_height())
            color = color_map[rectangle.get_label()]

            color = tuple(list(color * 255 for color in to_rgb(color)))
            label = rectangle.get_label()
            img = cv2.rectangle(
                img,
                (x, y),
                (x + width, y + height),
                color,
                3,
            )
            cv2.putText(
                img, label, (x - 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )
        cv2.imshow("My WebCam", img)

        # ESC key
        if cv2.waitKey(1) == 27:
            break






def test_youtube():
    video = pafy.new('https://youtu.be/m4KzIFQVN1w').getbest()
    detector = TfDefaultDetector("ssd_mobilenet_v2/fpnlite_320x320/1", "mscoco_label_map.pbtxt")
    detector.initialize()
    
    title = video.title
    
    # CV2 Setup
    cap = cv2.VideoCapture(video.url)
    fps = 15
    cap.set(fps, 0)
    while True:
        ret_val, img = cap.read()
        if ret_val is False:
            continue
        
        image = Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        predict = detector.predict(image, use_name=True)
        color_map = {
            item["name"]: color
            for item, color in zip(detector.category_map.values(), RANDOM_COLORS)
        }
        # predict_person = filter_predict(predict, 'person')
        predict_cut = cut_threshold(predict, 0.5)
        rectangles = get_object_patch(image, predict_cut)
        
        for rectangle in rectangles:
            x, y = list(int(xy) for xy in rectangle.get_xy())
            width = int(rectangle.get_width())
            height = int(rectangle.get_height())
            color = color_map[rectangle.get_label()]
            
            color = tuple(list(color * 255 for color in to_rgb(color)))
            label = rectangle.get_label()
            img = cv2.rectangle(
                img,
                (x, y),
                (x + width, y + height),
                color,
                3,
            )
            cv2.putText(
                img, label, (x - 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )
        cv2.imshow(title, img)
        
        # ESC key
        if cv2.waitKey(1) == 27:
            break



if __name__ == "__main__":
    # test_image()
    # test_video()
    test_youtube()