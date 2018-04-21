from enum import Enum

import cv2

from face_recognition_model import FaceRecognitionModel
from file_path_manager import FilePathManager
from server.handler import Handler


class FaceRecognitionHandler(Handler):

    def __init__(self, name):
        self.face_recognition = FaceRecognitionModel(name)
        self.images = []
        self.base_path = FilePathManager.resolve("saved_images")

    def handle(self, image, question, type, name):
        if type == Type.FACE_RECOGNITION:
            return {
                "result": self.face_recognition.predict(image)
            }
        if type == Type.ADD_PERSON:
            cv2.imwrite(f"{self.base_path}/image_{len(self.images) + 1}.jpg", image)
            self.images.append(image)
        if type == Type.END_ADD_PERSON:
            self.face_recognition.add_person(name, self.images)
            self.images = []
        if type == Type.REMOVE_PERSON:
            self.face_recognition.remove_person(name)
        return {
            "result": "success"
        }


class Type(Enum):
    FACE_RECOGNITION = "face-recognition"
    ADD_PERSON = "add-person"
    REMOVE_PERSON = "remove-person"
    END_ADD_PERSON = "end-add-person"
