import pickle

import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms

from file_path_manager import FilePathManager
from image_to_text.build_vocab import Vocabulary
from image_to_text.model import EncoderCNN, DecoderRNN


class ImageToTextModel:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    def __init__(self):
        with open(FilePathManager.resolve("image_to_text/data/vocab.pkl"), 'rb') as f:
            self.vocab: Vocabulary = pickle.load(f)
        encoder = EncoderCNN(256)
        encoder.eval()
        decoder = DecoderRNN(256,
                             512,
                             len(self.vocab),
                             1)
        encoder.load_state_dict(torch.load(FilePathManager.resolve("image_to_text/models/encoder-5-3000.pkl")))
        decoder.load_state_dict(torch.load(FilePathManager.resolve("image_to_text/models/decoder-5-3000.pkl")))
        encoder = encoder
        decoder = decoder
        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        self.encoder = encoder
        self.decoder = decoder

    def predict(self, image):
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ImageToTextModel.transform(image)
        image = image.float().unsqueeze(0)
        image = Variable(image)
        feature = self.encoder(image)
        sampled_ids = self.decoder.sample(feature)
        sampled_ids = sampled_ids.cpu().data.numpy()
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sampled_caption = sampled_caption[1:-1]
        sentence = ' '.join(sampled_caption)
        return sentence


if __name__ == '__main__':
    image_to_text = ImageToTextModel()
    image = cv2.imread(FilePathManager.resolve("image_to_text/test_images/test.png"))
    print(image_to_text.predict(image))
