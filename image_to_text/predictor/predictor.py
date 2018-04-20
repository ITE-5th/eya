from abc import abstractmethod, ABCMeta


class Predictor(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, image):
        raise NotImplementedError()
