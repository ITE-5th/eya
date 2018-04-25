import cv2

from file_path_manager import FilePathManager
from image_to_text.predictor.predictor import Predictor
from image_to_text.predictor.convcap_predictor import ConvcapPredictor
from image_to_text.predictor.encoder_decoder_predictor import EncoderDecoderPredictor


class ImageToTextModel:

    def __init__(self, model_type="convcap"):
        self.predictor: Predictor = None
        if model_type == "encoder-decoder":
            self.predictor = EncoderDecoderPredictor()
        elif model_type == "convcap":
            self.predictor = ConvcapPredictor()

    def predict(self, image):
        return self.predictor.predict(image)


if __name__ == '__main__':
    image_to_text = ImageToTextModel()
    image = cv2.imread(FilePathManager.resolve("image_to_text/test_images/test.png"))
    print(image_to_text.predict(image))
