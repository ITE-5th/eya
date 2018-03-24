from face_recognition.recognition.predictor.evm_predictor import EvmPredictor
from file_path_manager import FilePathManager


class FaceRecognitionModel:
    def __init__(self, user_name):
        try:
            self.model_path = FilePathManager.resolve(f"face_recognition/recognition/models/{user_name}/model.model")
            self.predictor = EvmPredictor(self.model_path)
        except Exception:
            raise FileNotFoundError("The model was not found")

    def add_person(self, person_name, images):
        self.predictor.add_person(person_name, images)

    def remove_person(self, person_name):
        self.predictor.remove_person(person_name)

    def predict(self, face):
        temp = self.predictor.predict_from_image(face)
        temp = [x[0] for x in temp]
        return ",".join(temp)
