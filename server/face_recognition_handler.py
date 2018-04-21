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
        if type == "face-recognition":
            return {
                "result": self.face_recognition.predict(image)
            }
        if type == "add-person":
            cv2.imwrite(f"{self.base_path}/image_{len(self.images) + 1}.jpg", image)
            self.images.append(image)
        if type == "end-add-person":
            self.face_recognition.add_person(name, self.images)
            self.images = []
        if type == "remove-person":
            self.face_recognition.remove_person(name)
        return {
            "result": "success"
        }
