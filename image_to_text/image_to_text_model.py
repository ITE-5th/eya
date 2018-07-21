from image_to_text.predictor.convolutional_caption_predictor import ConvolutionalCaptionPredictor


class ImageToTextModel:

    def __init__(self):
        self.predictor = ConvolutionalCaptionPredictor()

    @staticmethod
    def process_result(caption):
        caption = caption.replace("<unk>", " ")
        # caption = re.sub(r'(.+) \1+', r'\1', caption)
        return caption

    def predict(self, image):
        result = self.process_result(self.predictor.predict(image))
        return result
