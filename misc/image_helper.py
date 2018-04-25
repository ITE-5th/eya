import base64

import cv2
import numpy as np


class ImageHelper:
    @staticmethod
    def to_image(img_data):
        return cv2.imdecode(np.fromstring(base64.decodebytes(img_data.encode()), np.uint8), cv2.IMREAD_COLOR)
