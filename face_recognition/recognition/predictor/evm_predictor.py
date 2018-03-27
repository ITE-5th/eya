import joblib
import numpy as np

from recognition.estimator.evm import EVM
from recognition.predictor.predictor import Predictor
from recognition.preprocessing.image_feature_extractor import ImageFeatureExtractor


class EvmPredictor(Predictor):

    def __init__(self, evm_model_path: str):
        insight = True
        super().__init__(insight=insight)
        self.model_path = evm_model_path
        self.evm: EVM = joblib.load(self.model_path)

    def reload(self):
        self.evm = joblib.load(self.model_path)

    def save(self):
        joblib.dump(self.evm, self.model_path)

    def add_person(self, person_name, images):
        X = ImageFeatureExtractor.extract_from_images(images, insight=self.insight)
        y = np.full((len(images), 1), person_name)
        self.evm.fit(X, y)
        self.save()

    def remove_person(self, person_name):
        self.evm.remove(person_name)
        self.save()

    def predict_from_image(self, image):
        items = super().predict_from_image(image)
        result = []
        for (face, rect) in items:
            x = face.data.cpu().numpy().reshape(1, -1) if not self.insight else face.reshape(1, -1)
            predicted = self.evm.predict_with_prop(x)
            clz, prop = predicted[0]
            result.append((clz, rect, prop))
        return result
