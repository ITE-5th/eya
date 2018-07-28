import os
import warnings
from enum import Enum
from random import random
from shutil import copy2

import cv2
import joblib
import matplotlib.pyplot as plt

from face.predictor.evm_predictor import EvmPredictor
from face.predictor.similarity_predictor import SimilarityPredictor
from file_path_manager import FilePathManager

warnings.filterwarnings("ignore")


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

    @staticmethod
    def show_recognition_result(image, predicted):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.cla()
        plt.axis("off")
        plt.imshow(image)
        for (name, _, rect) in predicted:
            name = name.replace("_", " ").title()
            color = (random(), random(), random())
            x, y, w, h = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
            rect = plt.Rectangle((x, y),
                                 w,
                                 h,
                                 fill=False,
                                 edgecolor=color,
                                 linewidth=2.5)
            plt.gca().add_patch(rect)
            plt.gca().text(x + 15, y - 10,
                           name,
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
        plt.show()

    def predict(self, face):
        predicted = self.predictor.predict_from_image(face)
        if len(predicted) != 0:
            self.show_recognition_result(face, predicted)
            predicted = [f"{x[0]} {x[1]}" for x in predicted]
        return ",".join(predicted)


if __name__ == '__main__':
    model = FaceRecognitionModel("zaher")
    face = cv2.imread(FilePathManager.resolve(
        "face/test_faces/captured.jpg"))
    print(model.predict(face))
