from server.message.image_message import ImageMessage
from server.message.name_message import NameMessage


class AddPersonMessage(NameMessage, ImageMessage):
    def __init__(self, name, image):
        ImageMessage.__init__(self, image)
        NameMessage.__init__(self, name)
