from collections import Counter

import cv2
import inflect
from nltk.corpus import wordnet as wn

from file_path_manager import FilePathManager
from object_recognition.predictor.retina_net.retina_net_predictor import RetinaNetPredictor


class ObjectRecognitionModel:
    def __init__(self):
        self.predictor = RetinaNetPredictor()
        self.classes = set(self.predictor.classes)
        self.p = inflect.engine()

    def count_objects(self, counter):
        result = [f"{value} {self.p.plural(key, value)}" for key, value in counter.items()]
        return result

    @staticmethod
    def extract_name(syn):
        temp = syn.name()
        return temp[:temp.index(".")]

    def check_object(self, object_name):
        obj = wn.synsets(object_name)
        if not obj:
            return False
        obj = obj[0]
        hypo = set([object_name] + [self.extract_name(i) for i in obj.closure(lambda s: s.hyponyms())])
        return len(hypo.intersection(self.classes))

    def count_object(self, counter, object_name):
        if not self.check_object(object_name):
            return ["-1"]
        result = []
        obj = wn.synsets(object_name)[0]
        for key, value in counter.items():
            k = wn.synsets(key)[0]
            hyper = set([i for i in k.closure(lambda s: s.hypernyms())])
            if object_name == key or obj in hyper:
                result.append(f"{value} {self.p.plural(key, value)}")
        return result

    def predict(self, image, object_name=""):
        objects = self.predictor.predict(image)
        counter = Counter(objects)
        if object_name == "":
            result = self.count_objects(counter)
        else:
            object_name = object_name.strip()
            if self.p.singular_noun(object_name):
                object_name = self.p.singular_noun(object_name)
            object_name = object_name.strip()
            result = self.count_object(counter, object_name)
        result = ",".join(result)
        return result


if __name__ == '__main__':
    model = ObjectRecognitionModel()
    image = cv2.imread(FilePathManager.resolve("vqa/test_images/two_girls.jpg"))
    print(model.predict(image, object_name="people"))
