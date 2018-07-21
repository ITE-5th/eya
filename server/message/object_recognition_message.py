from server.message.image_message import ImageMessage


class ObjectRecognitionMessage(ImageMessage):
    def __init__(self, image=None, object_name=None):
        super().__init__(image=image)
        self.object_name = object_name
