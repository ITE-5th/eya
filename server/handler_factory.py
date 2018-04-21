from image_to_text_model import ImageToTextModel
from server.face_recognition_handler import FaceRecognitionHandler
from server.image_to_text_handler import ImageToTextHandler
from server.register_face_recognition_handler import RegisterFaceRecognitionHandler
from server.vqa_handler import VqaHandler
from vqa_model import VqaModel


class HandlerFactory:
    def __init__(self):
        self.vqa = VqaModel()
        self.image_to_text = ImageToTextModel()
        self.face_recognition_models = dict()

    def create(self, type, name=None):
        if type == "visual-question-answering":
            return VqaHandler(self.vqa)
        if type == "image-to-text":
            return ImageToTextHandler(self.image_to_text)
        if type == "register-face-recognition":
            return RegisterFaceRecognitionHandler()
        if type == "start-face-recognition":
            temp = FaceRecognitionHandler(name)
            self.face_recognition_models[name] = temp
            return temp
        if type in ["face-recognition", "add-person", "end-add-person", "remove-person"]:
            return self.face_recognition_models[name]
