from face_recognition.recognition.predictor.evm_predictor import EvmPredictor
from file_path_manager import FilePathManager
from recognition.predictor.insightface_predictor import InsightfacePredictor


class FaceRecognitionModel:
    def __init__(self, user_name, type="insight"):
        try:
            if type == "evm":
                self.model_path = FilePathManager.resolve(f"face_recognition/recognition/models/{user_name}/evm.model")
                self.predictor = EvmPredictor(self.model_path)
            else:
                self.model_path = FilePathManager.resolve(
                    f"face_recognition/recognition/models/{user_name}/insight.model")
                self.predictor = InsightfacePredictor(self.model_path)
        except Exception:
            raise FileNotFoundError("The model was not found")

    def add_person(self, person_name, images):
        self.predictor.add_person(person_name, images)

    def remove_person(self, person_name):
        self.predictor.remove_person(person_name)

    def predict(self, face):
        temp = self.predictor.predict_from_image(face)
        temp = [f"{x[0]} {x[2]}" for x in temp]
        return ",".join(temp)
