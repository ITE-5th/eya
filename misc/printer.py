import re

from server.message.end_add_person_message import EndAddPersonMessage
from server.message.message import Message
from server.message.object_recognition_message import ObjectRecognitionMessage
from server.message.remove_person_message import RemovePersonMessage
from server.message.vqa_message import VqaMessage


class Printer:
    @staticmethod
    def print(message: Message):
        name = message.__class__.__name__
        name = name[:name.index("Message")]
        name = Printer.convert(name)
        print("-" * 167)
        print(name)
        Printer.handle_type(message)

    @staticmethod
    def handle_type(message):
        if isinstance(message, VqaMessage):
            print(f"We Will Answer: {message.question}")
        if isinstance(message, EndAddPersonMessage):
            name = message.name.replace("_", " ").title()
            print(f"We Will Add {name}")
        if isinstance(message, RemovePersonMessage):
            name = message.name.replace("_", " ").title()
            print(f"We Will Remove {name}")
        if isinstance(message, ObjectRecognitionMessage):
            if message.object_name is not None and message.object_name != "":
                print(f"We Will Count {message.question}")

    @staticmethod
    def convert(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
