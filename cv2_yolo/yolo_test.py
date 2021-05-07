import cv2
import urllib3
from urllib.parse import urlparse
import os


cfg = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
weights = "https://pjreddie.com/media/files/yolov3.weights"
manager = urllib3.PoolManager()
CHUNK_SIZE = 100
model_path = dict()

for yolo_file in [cfg, weights]:

    file_name = os.path.basename(urlparse(yolo_file).path)

    if os.getcwd().endswith("cv2_yolo"):
        path = os.getcwd()
    else:
        path = os.getcwd() + "/cv2_yolo"
    path += "/data"
    file_path = f"{path}/{file_name}"

    if file_name in os.listdir(path):
        print(f"{file_name} already exists")

    else:
        print(f"start download {file_name}")
        file = manager.request("GET", yolo_file, preload_content=False)

        with open(file_path, "wb") as out:
            while True:
                data = file.read(CHUNK_SIZE)
                if not data:
                    break
            out.write(data)

    model_path[file_name] = file_path


dnn_model = cv2.dnn.readNet(model_path.get("yolov3.weigths"), model_path.get("yolov3.cfg"))
dnn_model