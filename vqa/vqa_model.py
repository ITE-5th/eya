import cv2

from file_path_manager import FilePathManager
from vqa.modified_model.predictor.predictor import Predictor


class VqaModel:
    def __init__(self):
        self.predictor = Predictor()

    def predict(self, question, image):
        temp = self.predictor.predict_from_image(question, image)
        temp = [x[0] for x in temp]
        return ",".join(temp)


if __name__ == '__main__':
    vqa = VqaModel()
    question = "what is the color of the umbrella?"
    image = cv2.imread(FilePathManager.resolve("vqa/test_images/girl_with_umbrella.jpg"))
    print(vqa.predict(question, image))