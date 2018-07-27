import math
import warnings

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import pyramid_expand

from file_path_manager import FilePathManager
from image_to_text.predictor.convolutional_caption_predictor import ConvolutionalCaptionPredictor

warnings.filterwarnings("ignore")


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
        words = caption
        regions = int(math.sqrt(regions))
        upscale = 224 / regions
        for t in range(len(words)):
            if t > 18:
                break
            plt.subplot(4, 5, t + 2)
            plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
            plt.imshow(image)
            alp_curr = alphas[t].view(regions, regions)
            alp_img = pyramid_expand(alp_curr.detach().numpy(), upscale=upscale, sigma=20)
            plt.imshow(alp_img, alpha=0.7)
            plt.axis('off')
        plt.show()

    @staticmethod
    def replace_repeating(caption):
        current_word = ""
        result = []
        for x in caption:
            if current_word != x:
                result.append(x)
                current_word = x
        return result

    @staticmethod
    def process_result(caption):
        caption = [x for x in caption if x != "<unk>"]
        caption = ImageToTextModel.replace_repeating(caption)
        if caption[len(caption) - 1] in ["and", "or", "a", "an", "the"]:
            caption = caption[:-1]
        return caption

    def predict(self, image):
        caption, attns = self.predictor.predict(image)
        self.show_attentions(image, caption, attns)
        caption = self.process_result(caption)
        caption = " ".join(caption)
        return caption


if __name__ == '__main__':
    model = ImageToTextModel()
    image = cv2.imread(FilePathManager.resolve("vqa/test_images/girl_with_umbrella.jpg"))
    print(model.predict(image))
