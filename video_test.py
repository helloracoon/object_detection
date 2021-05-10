import cv2
import glob
import os

from object_detection.detector import TfDefaultDetector
from object_detection.video import Capture, YoutubeCapture
from pathlib import Path

path = "/Users/helloracoon/Downloads/test.avi"

# download_youtube("https://youtu.be/VEA0a6gOydo", path)

if __name__ == "__main__":

    cap = YoutubeCapture(
        "https://youtu.be/VEA0a6gOydo",
        save_path=str((Path() / "test.avi").resolve()),
    )
    detector = TfDefaultDetector(
        "ssd_mobilenet_v2/fpnlite_320x320/1", "mscoco_label_map.pbtxt"
    )
    detector.initialize()
    # predict = cap.detection(detector, threshold=0.5, skips=5, filter_list=['person'])
    cap.detection_play(detector, threshold=0.5, skips=5, filter_list=['person'])
    # cap.detection_play(detector, threshold=0.5, skips=5, filter_list=None)



