import joblib
import numpy as np

from recognition.extractor.extractors import insight_extractor_forward
from recognition.predictor.predictor import Predictor
from recognition.preprocessing.image_feature_extractor import ImageFeatureExtractor


class InsightfacePredictor(Predictor):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.reload()

    def detect(self, features, threshold=1.24):
        features = features.reshape(1, -1)
        min_dist = 1e9
        max_per = None
        for (clz, class_features) in self.classes.items():
            distances = np.linalg.norm(class_features - features, axis=1, keepdims=True)
            dist = distances.min()
            if dist < min_dist:
                max_per = clz
                min_dist = dist
        return max_per if min_dist <= threshold else Predictor.UNKNOWN, min_dist

    def predict_from_image(self, image):
        items = ImageFeatureExtractor.aligner.preprocess_image(image)
        result = []
        for (face, rect) in items:
            features = insight_extractor_forward(face)
            clz, prop = self.detect(features)
            result.append((clz, rect, prop))
        return result

    def reload(self):
        self.classes = joblib.load(self.model_path)

    def save(self):
        joblib.dump(self.classes, self.model_path)

    def add_person(self, person_name, images):
        feats = []
        for image in images:
            image = ImageFeatureExtractor.aligner.preprocess_face_from_image(image)
            feat = insight_extractor_forward(image)
            feats.append(feat)
        self.classes[person_name] = np.array(feats)
        self.save()

    def remove_person(self, person_name):
        self.classes.pop(person_name)
        self.save()

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
