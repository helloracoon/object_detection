import numpy as np
import urllib3
import cv2
from urllib3.packages.six import BytesIO, StringIO

manager = urllib3.PoolManager()


class Image:
    def __init__(self, path, stream: bool = False):

        if stream:
            img_buffer = BytesIO(manager.request("GET", path).data)
            img_decoded = cv2.imdecode(
                np.frombuffer(img_buffer.read(), dtype=np.uint8), -1
            )
        else:
            img_decoded = cv2.imread(path)
        self.array = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)
        self.size = self.array.shape
