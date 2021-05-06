import time
from object_detection.functions import plot_object_detection, cut_threshold
from object_detection.detector import TfDefaultDetector
from object_detection.preprocess import Image

detector = TfDefaultDetector("ssd_mobilenet_v2/2", "mscoco_label_map.pbtxt")
detector.initialize()


def main():
    start = time.time()
    image = Image(
        "https://imgnews.pstatic.net/image/421/2021/05/06/0005334029_001_20210506113113527.jpg?type=w647",
        stream=True,
    )
    predict = detector.predict(
        image,
        use_name=True,
    )
    predict_cut = cut_threshold(predict, 0.6)
    ax = plot_object_detection(
        image,
        predict_cut,
        dpi=700
    )
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()


