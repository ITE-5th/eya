from collections import Counter

import cv2
import inflect
from nltk.corpus import wordnet as wn

from file_path_manager import FilePathManager
from object_recognition.predictor.retina_net.retina_net_predictor import RetinaNetPredictor


class ObjectRecognitionModel:
    def __init__(self):
        self.predictor = RetinaNetPredictor()
        self.p = inflect.engine()
        # self.word2vec = gensim.models.KeyedVectors.load(
        #     FilePathManager.resolve("object_recognition/data/word2vec.model"))

    def count_objects(self, counter):
        result = [f"{value} {self.p.plural(key, value)}" for key, value in counter.items()]
        return result

    def count_object(self, counter, object_name):
        result = []
        obj = wn.synsets(object_name)[0]
        for key, value in counter.items():
            k = wn.synsets(key)[0]
            hyper = set([i for i in k.closure(lambda s: s.hypernyms())])
            if object_name == key or obj in hyper:
                result.append(f"{value} {key}")
        result = ",".join(result)
        return result

    def predict(self, image, object_name=""):
        objects = self.predictor.predict(image)
        counter = Counter(objects)
        if object_name == "":
            result = self.count_objects(counter)
            result = ",".join(result)
        else:
            result = self.count_object(counter, object_name)
            result = str(result)
        return result


if __name__ == '__main__':
    model = ObjectRecognitionModel()
    image = cv2.imread(FilePathManager.resolve("vqa/test_images/girl.jpg"))
    print(model.predict(image, object_name="orange"))
