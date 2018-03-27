import cv2
import torch
import torch.nn as nn
from dlt.util.misc import cv2torch
from torch.autograd import Variable

from file_path_manager import FilePathManager
from recognition.estimator.insightface.face_embedding import FaceModel
from recognition.extractor.vgg_face import vgg_face


def remove_net(state):
    new_state = {}
    for key in state.keys():
        new_key = key[key.index(".") + 1:]
        new_state[new_key] = state[key]
    return new_state


vgg_extractor = vgg_face
# if not siamese:
state = torch.load(FilePathManager.resolve('face_recognition/data/VGG_FACE.pth'))
vgg_extractor.load_state_dict(state)
vgg_extractor = nn.Sequential(*list(vgg_extractor.children())[:-7])
# else:
#     state = torch.load(FilePathManager.resolve('data/VGG_FACE_MODIFIED.pth.tar'))
#     vgg_extractor = nn.Sequential(*list(vgg_extractor.children())[:-1])
#     state = state["state_dict"]
#     state = remove_net(state)
#     vgg_extractor.load_state_dict(state)
for param in vgg_extractor.parameters():
    param.requires_grad = False
vgg_extractor.eval()
# extractor = extractor.cuda()
insight_extractor = FaceModel(threshold=1.24, det=2, image_size="112,112",
                              model=FilePathManager.resolve("face_recognition/data/model-r50-am-lfw/model,0"))


def vgg_extractor_forward(x):
    return vgg_extractor.forward(x)


def insight_extractor_forward(x):
    return insight_extractor.get_feature(x)


if __name__ == '__main__':
    image = cv2.imread(FilePathManager.resolve("test_images/image_1.jpg"))
    image = cv2.resize(image, (200, 200))
    image = cv2torch(image).float()
    image = image.unsqueeze(0)
    image = Variable(image.cuda())
    print(vgg_extractor_forward(image))
