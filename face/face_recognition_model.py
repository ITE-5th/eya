import os
from enum import Enum
from shutil import copy2

import cv2
import joblib

from face.predictor.evm_predictor import EvmPredictor
from face.predictor.similarity_predictor import SimilarityPredictor
from file_path_manager import FilePathManager


class ModelType(Enum):
    EVM = 0
    SIMILARITY = 1


class FaceRecognitionModel:
    def __init__(self, user_name, type=ModelType.SIMILARITY):
        try:
            if type == ModelType.EVM:
                self.model_path = FilePathManager.resolve(
                    f"face/trained_models/{user_name}/base_model.model")
                self.predictor = EvmPredictor(self.model_path)
            elif type == ModelType.SIMILARITY:
                self.model_path = FilePathManager.resolve(
                    f"face/trained_models/{user_name}/similarity.model")
                self.predictor = SimilarityPredictor(self.model_path)
        except Exception:
            raise FileNotFoundError("The model was not found")

    @staticmethod
    def create_user(name, type=ModelType.SIMILARITY):
        FaceRecognitionModel.register(name)
        return FaceRecognitionModel(name, type)

    @staticmethod
    def register(name):
        path = FilePathManager.resolve("face/trained_models")
        base_model_path = f"{path}/base_model.model"
        person_path = f"{path}/{name}"
        if os.path.exists(person_path):
            return
        os.makedirs(person_path)
        evm_model_path = f"{person_path}/evm.model"
        copy2(base_model_path, evm_model_path)
        insight_model_path = f"{person_path}/similarity.model"
        joblib.dump({}, insight_model_path)

    def add_person(self, person_name, images):
        self.predictor.add_person(person_name, images)

    def remove_person(self, person_name):
        self.predictor.remove_person(person_name)

    def predict(self, face):
        temp = self.predictor.predict_from_image(face)
        temp = [f"{x[0]} {x[1]}" for x in temp]
        return ",".join(temp)


if __name__ == '__main__':
    model = FaceRecognitionModel("zaher")
    face = cv2.imread(FilePathManager.resolve("face/test_faces/brad-pitt-jennifer-aniston.jpg"))
    print(model.predict(face))
