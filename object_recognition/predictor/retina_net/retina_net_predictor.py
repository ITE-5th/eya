import numpy as np
import torch
from dlt.util import cv2torch
from torchvision.transforms import transforms

from file_path_manager import FilePathManager
from object_recognition.predictor.predictor import Predictor
from object_recognition.predictor.retina_net import model
from object_recognition.transforms.normalizer import Normalizer
from object_recognition.transforms.resizer import Resizer


class RetinaNetPredictor(Predictor):

    def __init__(self):
        self.classes = RetinaNetPredictor.load_class_names(
            FilePathManager.resolve("object_recognition/data/coco.names"))
        self.transform = transforms.Compose([Normalizer(), Resizer()])
        self.model = model.resnet50(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(FilePathManager.resolve("object_recognition/models/coco_resnet_50.pt")))
        self.model = self.model.cuda()
        self.model.eval()

    @staticmethod
    def load_class_names(path):
        with open(path, 'r') as fp:
            return [line.rstrip() for line in fp.readlines()]

    def convert_image(self, image):
        image = image.astype(np.float32) / 255.0
        image = {"img": image, "annot": np.array([])}
        image = self.transform(image)
        image = image["img"]
        image = cv2torch(image)
        image = image.unsqueeze(0)
        return image

    def predict(self, image):
        image = self.convert_image(image)
        image = image.float().cuda()
        with torch.no_grad():
            scores, classification, transformed_anchors = self.model(image)
            idxs = np.where(scores > 0.5)
            labels = [self.classes[int(classification[idxs[0][i]])] for i in range(idxs[0].shape[0])]
            return labels
