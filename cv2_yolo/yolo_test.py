import cv2
import urllib3
from urllib.parse import urlparse
import os
import numpy as np


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


coco_names = manager.request('GET','https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names', preload_content=False)
classes = [coco.decode('utf-8').rstrip() for coco in coco_names.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(model_path.get("yolov3.weigths"), model_path.get("yolov3.cfg"))
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
img = cv2.imread("/Users/helloracoon/Downloads/AKR20201207106900005_01_i_P4.jpg")
height, width, channels = img.shape
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)


#
# dnn_model = cv2.dnn.readNet(model_path.get("yolov3.weigths"), model_path.get("yolov3.cfg"))
# layer_names = dnn_model.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in dnn_model.getUnconnectedOutLayers()]
#
#
#
#
# #
# if __name__ == '__main__':
#
#     image = cv2.imread(filename='/Users/helloracoon/Downloads/AKR20201207106900005_01_i_P4.jpg')
#     blob = cv2.dnn.blobFromImage(image)
#     dnn_model.setInput(blob)
#     dnn_model.forward(output_layers)
