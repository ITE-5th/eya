import cv2

from file_path_manager import FilePathManager
from predictor.retina_net.retina_net_predictor import RetinaNetPredictor


class ObjectRecognitionModel:
    def __init__(self, unique_objects=True, use_gpu=True):
        self.unique_objects = unique_objects
        self.predictor = RetinaNetPredictor(use_gpu)

    def process_result(self, result):
        if self.unique_objects:
            result = list(set(result))
        return result

    def predict(self, image):
        result = self.predictor.predict(image)
        result = self.process_result(result)
        return ",".join(result)


if __name__ == '__main__':
    model = ObjectRecognitionModel(use_gpu=True)
    image = cv2.imread(FilePathManager.resolve("vqa/test_images/girl.jpg"))
    print(model.predict(image))
