from server.message.message import Message


class ImageMessage(Message):
    def __init__(self, image=None):
        self.image = image
