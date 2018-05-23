from server.message.message import Message


class NameMessage(Message):
    def __init__(self, name=None):
        self.name = name
        if name is not None:
            self.name = self.name.lower().replace(" ", "_")
