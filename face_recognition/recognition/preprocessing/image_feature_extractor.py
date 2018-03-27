import glob
import os

import cv2
import numpy as np
import torch
from dlt.util import cv2torch
from torch.autograd import Variable

from file_path_manager import FilePathManager
from recognition.extractor.extractors import vgg_extractor_forward, insight_extractor_forward
from recognition.preprocessing.aligner_preprocessor import AlignerPreprocessor


class ImageFeatureExtractor:
    aligner = AlignerPreprocessor(scale=1)

    @staticmethod
    def extract_from_images(images, insight=False):
        result = []
        for image in images:
            aligned = ImageFeatureExtractor.aligner.preprocess_face_from_image(image)
            if not insight:
                aligned = cv2torch(aligned).float().unsqueeze(0).cuda()
                aligned = vgg_extractor_forward(Variable(aligned))
                aligned = aligned.view(-1).cpu().data.numpy()
            else:
                aligned = cv2.resize(aligned, (112, 112))
                aligned = insight_extractor_forward(aligned)
            result.append(aligned)
        return np.asarray(result)

    @staticmethod
    def extract_from_dir(root_dir: str, insight=False):
        names = sorted(os.listdir(root_dir + "/custom_images2"))
        if not os.path.exists(root_dir + "/custom_features"):
            os.makedirs(root_dir + "/custom_features")
        for i in range(len(names)):
            name = names[i]
            path = root_dir + "/custom_images2/" + name
            if not os.path.exists(root_dir + "/custom_features/" + name):
                os.makedirs(root_dir + "/custom_features/" + name)
            faces = os.listdir(path)
            for face in faces:
                p = path + "/" + face
                image = cv2.imread(p)
                if not insight:
                    image = cv2.resize(image, (224, 224))
                    image = np.swapaxes(image, 0, 2)
                    image = np.swapaxes(image, 1, 2)
                    image = torch.from_numpy(image.astype(np.float)).float().unsqueeze(0).cuda()
                    image = vgg_extractor_forward(Variable(image))
                    image = image.view(-1).cpu()
                    res = (image.data, i)
                else:
                    image = cv2.resize(image, (112, 112))
                    feats = insight_extractor_forward(image)
                    res = (feats, i)
                temp = root_dir + "/custom_features/" + name + "/" + face[
                                                                     :face.rfind(
                                                                         ".")] + ".features"
                torch.save(res, temp)

    @staticmethod
    def load(root_dir: str):
        temp = sorted(glob.glob(root_dir + "/custom_features/**/*.features"))
        return [torch.load(face) for face in temp]


if __name__ == '__main__':
    ImageFeatureExtractor.extract_from_dir(FilePathManager.resolve("data"))
