from face_recognition_model import FaceRecognitionModel
from server.handler import Handler


class RegisterFaceRecognitionHandler(Handler):
    def handle(self, image, question, type, name):
        FaceRecognitionModel.register(name, remove_dir=False)
        result = {"result": "success"}
        return result
