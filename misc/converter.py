from server.message.add_person_message import AddPersonMessage
from server.message.close_message import CloseMessage
# from server.message.close_message import CloseMessage
from server.message.end_add_person_message import EndAddPersonMessage
from server.message.face_recognition_message import FaceRecognitionMessage
from server.message.image_message import ImageMessage
from server.message.image_to_text_message import ImageToTextMessage
from server.message.register_face_recognition_message import RegisterFaceRecognitionMessage
from server.message.remove_person_message import RemovePersonMessage
from server.message.start_face_recognition_message import StartFaceRecognitionMessage
from server.message.vqa_message import VqaMessage


class Converter(object):

    @staticmethod
    def to_object(message, from_json: bool = False):
        if not from_json:
            class_name = message.__class__.__name__
            class_ = globals()[class_name]
            obj = class_()
            obj.__dict__ = message.__dict__
            return obj

        raise AttributeError("Not Supported.")
