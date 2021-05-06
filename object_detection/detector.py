import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import os
from google.protobuf import text_format
from .preprocess import Image
from typing import Union


class Label(dict):
    def __init__(self, **kwargs):
        if "id" not in kwargs:
            raise ValueError("id is required")
        super().__init__(**kwargs)

    def sorted_item(self):
        return tuple(sorted(self.items()))

    def __hash__(self):
        return hash(self.sorted_item())

    def __lt__(self, other):
        return self.get("id") < other.get("id")


# customized tensorflow model  (https://github.com/tensorflow/models/tree/master/research/object_detection)
class TfDefaultDetector:

    _model_path = None
    _label_path = None
    _model = None
    category_map = dict()

    def __init__(self, model_name, label_name):
        self.model_name = model_name
        self._label_path = tf.keras.utils.get_file(
            fname=label_name,
            origin=f"https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/{label_name}",
            untar=False,
            cache_subdir=f"datasets/{model_name}",
        )

    def initialize(self):
        self._load_model()
        self._load_label()

    def _load_model(self):
        self._model = hub.load(
            f"https://hub.tensorflow.google.cn/tensorflow/{self.model_name}"
        )

    def _load_label(self) -> list:

        try:
            from .proto.string_int_label_map_pb2 import StringIntLabelMap, LVISFrequency
        except ModuleNotFoundError:
            from . import proto

            proto_dir = os.path.dirname(proto.__file__)
            proto_name = "string_int_label_map.proto"
            os.system(
                f"protoc --proto_path={proto_dir} --python_out={proto_dir} {proto_name}"
            )
            from .proto.string_int_label_map_pb2 import StringIntLabelMap, LVISFrequency

        with tf.io.gfile.GFile(self._label_path, "r") as fid:
            label_map_string = fid.read()
            label_map = StringIntLabelMap()
            try:
                text_format.Merge(label_map_string, label_map)
            except text_format.ParseError:
                label_map.ParseFromString(label_map_string)

        max_num_classes = max(item.id for item in label_map.item)
        categories = set()

        for item in label_map.item:
            if not 0 < item.id <= max_num_classes:
                continue
            if item.HasField("display_name"):
                name = item.display_name
            else:
                name = item.name
            category = Label(id=item.id, name=name)
            if item.HasField("frequency"):
                if item.frequency == LVISFrequency.Value("FREQUENT"):
                    category["frequency"] = "f"
                elif item.frequency == LVISFrequency.Value("COMMON"):
                    category["frequency"] = "c"
                elif item.frequency == LVISFrequency.Value("RARE"):
                    category["frequency"] = "r"
            if item.HasField("instance_count"):
                category["instance_count"] = item.instance_count
            if item.keypoints:
                keypoints = {}
                list_of_keypoint_ids = []
                for kv in item.keypoints:
                    if kv.id in list_of_keypoint_ids:
                        raise ValueError(
                            "Duplicate keypoint ids are not allowed. "
                            "Found {} more than once".format(kv.id)
                        )
                    keypoints[kv.label] = kv.id
                    list_of_keypoint_ids.append(kv.id)
                category["keypoints"] = keypoints
            categories.add(category)
        self.category_map = dict(sorted(
            {item.pop("id"): item for item in categories}.items()
        ))
        return list(categories)

    def predict(self, image: Image, use_name: bool = False) -> dict:

        input_tensor = tf.convert_to_tensor(image.array)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = self._model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections

        # detection_classes should be ints.
        detections["detection_classes"] = detections["detection_classes"].astype(
            np.int64
        )
        if use_name:
            detections["detection_classes"] = [
                self.category_map[cls] for cls in detections["detection_classes"]
            ]
        return detections
