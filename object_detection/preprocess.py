import numpy as np
import requests
from PIL import Image as img


class Image:
    def __init__(self, path, stream: bool = False):
        if stream:
            image = requests.get(path, stream=True).raw
        else:
            image = path
        self.pil_image = img.open(image)
        self.array = np.array(self.pil_image)
