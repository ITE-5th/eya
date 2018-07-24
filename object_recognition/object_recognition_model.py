import os
from collections import Counter
from random import random

import cv2
import inflect
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import wordnet as wn

from file_path_manager import FilePathManager
from object_recognition.predictor.retina_net.retina_net_predictor import RetinaNetPredictor
from transforms.unnormalizer import UnNormalizer


class ObjectRecognitionModel:
    def __init__(self, vis=True):
        self.predictor = RetinaNetPredictor()
        self.classes = set(self.predictor.classes)
        self.p = inflect.engine()
        self.vis = vis

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

    @staticmethod
    def show_prediction(image, objects, anchors, idxs):
        unnormalize = UnNormalizer()
        image = np.array(255 * unnormalize(image[0, :, :, :])).copy()
        image[image < 0] = 0
        image[image > 255] = 255
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        temp = "temp.png"
        cv2.imwrite(temp, image)
        image = mpimg.imread(temp)
        plt.cla()
        plt.axis("off")
        plt.imshow(image)
        for j in range(idxs[0].shape[0]):
            color = (random(), random(), random())
            bbox = anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = objects[j]
            rect = plt.Rectangle((x1, y1),
                                 abs(x2 - x1),
                                 abs(y2 - y1),
                                 fill=False,
                                 edgecolor=color,
                                 linewidth=2.5)
            plt.gca().add_patch(rect)
            plt.gca().text(x1 + 3, y1 - 10,
                           label_name.title(),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
        plt.show()
        os.remove(temp)

    def predict(self, image, object_name=""):
        objects, image, anchors, idxs = self.predictor.predict(image)
        if self.vis:
            self.show_prediction(image, objects, anchors, idxs)
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
    image = cv2.imread(FilePathManager.resolve("vqa/test_images/doing.jpg"))
    print(model.predict(image, object_name=""))
