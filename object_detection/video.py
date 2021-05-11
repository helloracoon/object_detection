import cv2
import pafy
import os

from matplotlib.colors import to_rgb

from object_detection.detector import TfDefaultDetector
from object_detection.functions import (
    cut_threshold,
    get_object_patch,
    RANDOM_COLORS,
    filter_predict,
)
from object_detection.preprocess import Image


class Capture:
    def __init__(self, path):
        self.path = path
        self.vid = cv2.VideoCapture(self.path)
        fps = self.vid.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / fps)

    def _play(self, skips):
        skipper = 0
        while self.vid.isOpened():
            retval, image = self.vid.read()
            if retval:
                skipper += 1
                if skipper % skips != 0:
                    continue
                yield image

    def play(self, skips=1):

        for image in self._play(skips):
            cv2.imshow(self.path, image)
            if cv2.waitKey(self.delay) & 0xFF == ord("q"):
                break

    def detection(
        self,
        detector: TfDefaultDetector,
        threshold: float = 0,
        skips: int = 1,
        filter_list: list = None,
    ) -> dict:

        for image in self._play(skips):
            image = Image(image)
            time = self.vid.get(cv2.CAP_PROP_POS_MSEC)
            predict = detector.predict(image, use_name=True)
            predict = filter_predict(predict, filter_list)
            predict = cut_threshold(predict, threshold)
            if not predict or not predict["detection_classes"]:
                continue
            yield {time: predict}

    def detection_play(
        self,
        detector: TfDefaultDetector,
        threshold: float = 0,
        skips: int = 1,
        filter_list: list = None,
    ) -> None:
        color_map = {
            item["name"]: color
            for item, color in zip(detector.category_map.values(), RANDOM_COLORS)
        }

        for image in self._play(skips):

            image = Image(image)
            predict = detector.predict(image, use_name=True)

            predict = filter_predict(predict, filter_list)
            predict_cut = cut_threshold(predict, threshold)
            rectangles = get_object_patch(image, predict_cut)
            img = image.array
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
                    img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                )
            cv2.imshow(str(self.path), img)
            if cv2.waitKey(self.delay) & 0xFF == ord("q"):
                break

    def __enter__(self):

        for image in self._play():
            yield image

        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.vid.release()


class YoutubeCapture(Capture):
    def __init__(self, path, save_path):
        if os.path.isfile(save_path):
            print("file already exist")
        else:
            video = pafy.new(path)
            resolutions = sorted(
                [video for video in video.allstreams if str(video).startswith("video")],
                key=lambda x: int(str(x).split("x")[-1]),
                reverse=False,
            )
            for index, resolution in enumerate(resolutions):
                print(f"[{index}] {resolution}")

            selected = input(f"select resolution [0-{len(resolutions)}] :")
            try:
                selected = int(selected)
            except ValueError:
                raise ValueError("Wrong Type Error")
            download = resolutions[selected].download(filepath=save_path)
        super().__init__(save_path)
