import time

import cv2
import glob
import os

from object_detection.detector import TfDefaultDetector
from object_detection.video import Capture, YoutubeCapture
from pathlib import Path

path = "/Users/helloracoon/Downloads/test.avi"
detector = TfDefaultDetector(
    "ssd_mobilenet_v2/fpnlite_320x320/1", "mscoco_label_map.pbtxt"
)
detector.initialize()

cap = YoutubeCapture(
    "https://youtu.be/VEA0a6gOydo", save_path=str((Path() / "test.avi").resolve())
)



def detect():
    for predict in cap.detection(detector=detector, threshold=0.5, skips=10):
        time.sleep(1)
        print(predict)


def play():
    cap.play()

    
def play_detect(**kwargs):
    cap.detection_play(**kwargs)

def context_form():
    with YoutubeCapture(
            "https://youtu.be/VEA0a6gOydo", save_path=str((Path() / "test.avi").resolve())
    ) as c:
        for v in c:
            print(v)
        

        
if __name__ == "__main__":
    # context_form()
    # play()
    # test_predict()
    play_detect(detector=detector, threshold=0.5, skips=10)
    pass
