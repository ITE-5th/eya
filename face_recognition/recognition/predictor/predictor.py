import os
from abc import ABCMeta, abstractmethod

import cv2
from dlt.util.misc import cv2torch
from torch.autograd import Variable

from file_path_manager import FilePathManager
from recognition.extractor.extractors import vgg_extractor
from recognition.preprocessing.aligner_preprocessor import AlignerPreprocessor
from recognition.preprocessing.image_feature_extractor import ImageFeatureExtractor


class Predictor(metaclass=ABCMeta):

    def predict_from_path(self, image_path: str):
        return self.predict_from_image(cv2.imread(image_path))

    @abstractmethod
    def predict_from_image(self, image):
        items = ImageFeatureExtractor.aligner.preprocess_image(image)
        result = []
        for (face, rect) in items:
            face = cv2.resize(face, (200, 200))
            face = cv2torch(face).float()
            face = face.unsqueeze(0)
            x = Variable(face).cuda()
            x = ImageFeatureExtractor.extractor(x)
            result.append((x, rect))
        return result
