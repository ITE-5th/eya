from collections import Counter

import cv2
import inflect

from file_path_manager import FilePathManager
from object_recognition.predictor.retina_net.retina_net_predictor import RetinaNetPredictor


class ObjectRecognitionModel:
    def __init__(self):
        self.predictor = RetinaNetPredictor()
        self.p = inflect.engine()

    def process_result(self, result):
        counter = Counter(result)
        result = [f"{value} {self.p.plural(key, value)}" for key, value in counter.items()]
        return result

    def predict(self, image):
        result = self.predictor.predict(image)
        result = self.process_result(result)
        return ",".join(result)


if __name__ == '__main__':
    model = ObjectRecognitionModel()
    image = cv2.imread(FilePathManager.resolve("vqa/test_images/doing.jpg"))
    print(model.predict(image))
