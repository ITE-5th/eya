import pickle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms, Scale

from convcap.beamsearch import beamsearch
from convcap.convcap import convcap
from convcap.vggfeats import Vgg16Feats
from file_path_manager import FilePathManager
from image_to_text.predictor.predictor import Predictor


class ConvcapPredictor(Predictor):
    ts = transforms.Compose([
        Scale([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, beam_size: int = 5):
        self.max_tokens = 15
        self.beam_size = beam_size
        num_layers = 3
        worddict_tmp = pickle.load(open(FilePathManager.resolve('image_to_text/data/wordlist.p'), 'rb'))
        wordlist = [l for l in iter(list(worddict_tmp.keys())) if l != '</S>']
        wordlist = ['EOS'] + sorted(wordlist)
        numwords = len(wordlist)
        self.wordlist = wordlist

        model_imgcnn = Vgg16Feats()
        model_imgcnn.cuda()

        model_convcap = convcap(numwords, num_layers, is_attention=True)
        model_convcap.cuda()
        checkpoint = torch.load(FilePathManager.resolve("image_to_text/models/convcap-model.pth"))
        model_convcap.load_state_dict(checkpoint['state_dict'])
        model_imgcnn.load_state_dict(checkpoint['img_state_dict'])

        model_imgcnn.train(False)
        model_convcap.train(False)
        self.model_imgcnn = model_imgcnn
        self.model_convcap = model_convcap

    @staticmethod
    def convert_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ConvcapPredictor.ts(image)
        image = image.unsqueeze(0)
        return image

    def predict(self, image):
        image = self.convert_image(image)
        img_v = Variable(image.cuda())
        imgfeats, imgfc7 = self.model_imgcnn(img_v)

        b, f_dim, f_h, f_w = imgfeats.size()
        imgfeats = imgfeats.unsqueeze(1).expand(b, self.beam_size, f_dim, f_h, f_w)
        imgfeats = imgfeats.contiguous().view(b * self.beam_size, f_dim, f_h, f_w)

        b, f_dim = imgfc7.size()
        imgfc7 = imgfc7.unsqueeze(1).expand(b, self.beam_size, f_dim)
        imgfc7 = imgfc7.contiguous().view(b * self.beam_size, f_dim)

        beam_searcher = beamsearch(self.beam_size, 1, self.max_tokens)

        wordclass_feed = np.zeros((self.beam_size * 1, self.max_tokens), dtype='int64')
        wordclass_feed[:, 0] = self.wordlist.index('<S>')
        outcaps = np.empty((1, 0)).tolist()

        for j in range(self.max_tokens - 1):
            wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

            wordact, attn = self.model_convcap(imgfeats, imgfc7, wordclass)
            wordact = wordact[:, :, :-1]
            wordact_j = wordact[..., j]

            beam_indices, wordclass_indices = beam_searcher.expand_beam(wordact_j)

            if len(beam_indices) == 0 or j == (self.max_tokens - 2):
                generated_captions = beam_searcher.get_results()
                for k in range(1):
                    g = generated_captions[:, k]
                    outcaps[k] = [self.wordlist[x] for x in g]
            else:
                wordclass_feed = wordclass_feed[beam_indices]
                imgfc7 = imgfc7.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
                imgfeats = imgfeats.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
                for i, wordclass_idx in enumerate(wordclass_indices):
                    wordclass_feed[i, j + 1] = wordclass_idx

        num_words = len(outcaps[0])
        if 'EOS' in outcaps[0]:
            num_words = outcaps[0].index('EOS')
        outcap = ' '.join(outcaps[0][:num_words])
        return outcap
