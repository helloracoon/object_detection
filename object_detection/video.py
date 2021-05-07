import cv2


def capture(video):
    vid = cv2.VideoCapture(video)

    while vid.isOpened():
        ret, frame = vid.read()
