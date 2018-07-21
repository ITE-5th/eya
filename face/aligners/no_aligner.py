from face.aligners.base_aligner import BaseAligner
from face.bases.pipeline import Pipeline
from face.transforms.crop import Crop
from face.transforms.scale import Scale


class NoAligner(BaseAligner):
    def __init__(self, scale: float = 0.0):
        """
        :param scale:  between 0 (0%) and 1 (100%)
        """
        self.scale = scale
        self.pipeline = Pipeline([Scale(scale), Crop()])

    def forward(self, inputs):
        return self.pipeline(inputs)
