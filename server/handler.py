from abc import abstractmethod, ABCMeta


class Handler(metaclass=ABCMeta):

    @abstractmethod
    def handle(self, image, question, type, name):
        raise NotImplementedError()
