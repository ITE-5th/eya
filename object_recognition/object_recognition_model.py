from predictor.retina_net.retina_net_predictor import RetinaNetPredictor


class ObjectRecognitionModel:
    def __init__(self):
        self.predictor = RetinaNetPredictor()

    def predict(self, image):
        result = self.predictor.predict(image)
        return ",".join(result)
