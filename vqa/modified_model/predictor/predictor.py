import os
import pickle

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from file_path_manager import FilePathManager
from vqa.modified_model.image_features_extractor import ImageFeaturesExtractor
from vqa.modified_model.net.base_model import build_baseline0_newatt_test
from vqa.modified_model.net.dataset import Dictionary, VQAFeatureDataset


def remove_module(state):
    new_state = {}
    for key, val in state.items():
        new_state[key[key.index(".") + 1:]] = val
    return new_state


def modify_state_dict(state):
    new_state = {}
    for k, v in state.items():
        if k != "v_att.linear.bias" and len(v.size()) == 1 and v.size(0) == 1:
            new_state[k] = torch.tensor(state[k][0])
        else:
            new_state[k] = state[k]
    return new_state


use_cuda = False
label2ans = pickle.load(open(FilePathManager.resolve("vqa/modified_model/data/trainval_label2ans.pkl"), 'rb'))
dictionary = Dictionary.load_from_file(FilePathManager.resolve('vqa/modified_model/data/dictionary.pkl'))
models = []
model_path = FilePathManager.resolve("vqa/modified_model/stored_models")
paths = ["{}/{}".format(model_path, path) for path in os.listdir(model_path)]
for path in paths:
    checkpoint = torch.load(path)
    ntokens, num_answers, v_dim, num_hidden = checkpoint["ntokens"], checkpoint["num_ans_candidates"], checkpoint[
        "v_dim"], 1024
    net = build_baseline0_newatt_test(ntokens, v_dim, num_answers, num_hidden)
    state = remove_module(checkpoint["state_dict"])
    net.load_state_dict(modify_state_dict(state))
    net.eval()
    if use_cuda:
        net = net.cuda()
    for param in net.parameters():
        param.requires_grad = False
    models.append(net)


class Predictor:
    def __init__(self, k=3):
        self.k = k

    def predict_from_image(self, question, image):
        try:
            padding = VQAFeatureDataset.pad_question(question, dictionary)
            spatial_features, image_features = ImageFeaturesExtractor.extract_from_image(image)
            question = Variable(torch.LongTensor(padding).unsqueeze(0))
            image_features = Variable(torch.FloatTensor(image_features).unsqueeze(0))
            spatial_features = Variable(torch.FloatTensor(spatial_features).unsqueeze(0))
            result = self.predict_from_models(image_features, spatial_features, question)
            values, indices = result.topk(self.k)
            return [(str(label2ans[i]), value.item()) for i, value in zip(indices, values)]
        except Exception:
            return "I can't answer this question because it contains some unknown words for me"

    @staticmethod
    def predict_from_models(image_features, spatial_features, question):
        result = torch.zeros(1, num_answers)
        for i in range(len(models)):
            model = models[i]
            result.add_(F.sigmoid(model(image_features, spatial_features, question).data))
        result = result / len(models)
        return result[0]
