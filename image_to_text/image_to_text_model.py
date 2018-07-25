import math

import cv2
import matplotlib.pyplot as plt
import skimage
from PIL import Image

from image_to_text.predictor.convolutional_caption_predictor import ConvolutionalCaptionPredictor


class ImageToTextModel:

    def __init__(self):
        self.predictor = ConvolutionalCaptionPredictor()

    @staticmethod
    def show_attentions(image, caption, alphas, regions=49):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize([224, 224], Image.LANCZOS)
        plt.subplot(4, 5, 1)
        plt.imshow(image)
        plt.axis('off')
        words = caption.split(" ")
        regions = int(math.sqrt(regions))
        upscale = 224 / regions
        for t in range(len(words)):
            if t > 18:
                break
            plt.subplot(4, 5, t + 2)
            plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
            plt.imshow(image)
            alp_curr = alphas[t].view(regions, regions)
            alp_img = skimage.transform.pyramid_expand(alp_curr.detach().numpy(), upscale=upscale, sigma=20)
            plt.imshow(alp_img, alpha=0.7)
            plt.axis('off')
        plt.show()

    @staticmethod
    def process_result(caption):
        caption = caption.replace("<unk>", " ")
        # caption = re.sub(r'(.+) \1+', r'\1', caption)
        return caption

    def predict(self, image):
        caption, attns = self.predictor.predict(image)
        # self.show_attentions(image, caption, attns)
        caption = self.process_result(caption)
        return caption
