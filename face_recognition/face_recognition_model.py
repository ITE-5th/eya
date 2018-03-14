import cv2

from face_recognition.recognition.predictor.evm_predictor import EvmPredictor
from file_path_manager import FilePathManager


class FaceRecognitionModel:
    def __init__(self):
        self.predictor = EvmPredictor(FilePathManager.resolve("face_recognition/recognition/models/evm.model"))

    def predict(self, face):
        temp = self.predictor.predict_from_image(face)
        temp = [x[0] for x in temp]
        return ",".join(temp)


if __name__ == '__main__':
    face_model = FaceRecognitionModel()
    path = FilePathManager.resolve("face_recognition/test_images/image_3.jpg")
    image = cv2.imread(path)
    print(face_model.predict(image))
