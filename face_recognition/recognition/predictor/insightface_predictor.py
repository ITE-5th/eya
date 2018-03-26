import joblib
import numpy as np

from file_path_manager import FilePathManager
from recognition.estimator.insightface.face_embedding import FaceModel
from recognition.predictor.predictor import Predictor
from recognition.preprocessing.image_feature_extractor import ImageFeatureExtractor


class InsightfacePredictor(Predictor):
    def __init__(self, model_path: str):
        super().__init__()
        self.face_model = FaceModel(threshold=1.24, det=2, image_size="112,112",
                                    model=FilePathManager.resolve("face_recognition/data/model-r50-am-lfw/model,0"))
        self.model_path = model_path
        self.reload()

    def detect(self, features, threshold=0.4):
        features = features.reshape(1, -1)
        features = InsightfacePredictor.normalize(features)
        max_sim = -1
        max_per = None
        for (clz, all_class_features) in self.classes.items():
            for feat in all_class_features:
                sim = np.dot(features, feat)
                if sim > max_sim:
                    max_per = clz
                    max_sim = sim
        return max_per if max_sim >= threshold else Predictor.UNKNOWN, max_sim

    def predict_from_image(self, image):
        print(self.classes)
        items = ImageFeatureExtractor.aligner.preprocess_image(image)
        result = []
        for (face, rect) in items:
            features = self.face_model.get_feature(face)
            clz, prop = self.detect(features)
            result.append((clz, rect, prop))
        return result

    def reload(self):
        self.classes = joblib.load(self.model_path)

    def save(self):
        joblib.dump(self.classes, self.model_path)

    def add_person(self, person_name, images):
        self.classes[person_name] = []
        for image in images:
            image = ImageFeatureExtractor.aligner.preprocess_face_from_image(image)
            feats = self.face_model.get_feature(image)
            feats = InsightfacePredictor.normalize(feats)
            feats = feats.reshape(-1, 1)
            self.classes[person_name].append(feats)
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
