import re

from image_to_text.predictor.convcap_predictor import ConvcapPredictor
from image_to_text.predictor.encoder_decoder_predictor import EncoderDecoderPredictor
from image_to_text.predictor.predictor import Predictor


class ImageToTextModel:

    def __init__(self, model_type="convcap"):
        self.predictor: Predictor = None
        if model_type == "encoder-decoder":
            self.predictor = EncoderDecoderPredictor()
        elif model_type == "convcap":
            self.predictor = ConvcapPredictor()

    @staticmethod
    def process_result(caption):
        caption = caption.replace("<unk>", " ")
        return re.sub(r'(.)\1+', r'\1', caption)

    def predict(self, image):
        result = self.process_result(self.predictor.predict(image))
        return result
