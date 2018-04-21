from enum import Enum

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
        if type == Type.VQA:
            return VqaHandler(self.vqa)
        if type == Type.ITT:
            return ImageToTextHandler(self.image_to_text)
        if type == Type.RFR:
            return RegisterFaceRecognitionHandler()
        if type in [Type.SFR, Type.AP, Type.EAP, Type.RP]:
            if name in self.face_recognition_models:
                return self.face_recognition_models[name]
            temp = FaceRecognitionHandler(name)
            self.face_recognition_models[name] = temp
            return temp


class Type(Enum):
    VQA = "visual-question-answering"
    ITT = "image-to-text"
    RFR = "register-face-recognition"
    SFR = "start-face-recognition"
    AP = "add-person"
    EAP = "end-add-person"
    RP = "remove-person"
