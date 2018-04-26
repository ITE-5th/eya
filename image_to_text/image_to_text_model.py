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

    def predict(self, image):
        return self.predictor.predict(image)
