import base64

import cv2
import numpy as np


class Converter(object):

    @staticmethod
    def to_object(message, json: bool = False):
        if json:
            class_name = message['_type']
        else:
            class_name = message.__class__.__name__
            message = message.__dict__
        class_ = globals()[class_name]
        obj = class_()
        obj.__dict__ = message
        return obj

    @staticmethod
    def to_image(img_data):
        return cv2.imdecode(np.fromstring(base64.decodebytes(img_data.encode()), np.uint8), cv2.IMREAD_COLOR)

#  from server.message.add_person_message import AddPersonMessage
# from server.message.close_message import CloseMessage
# from server.message.end_add_person_message import EndAddPersonMessage
# from server.message.face_recognition_message import FaceRecognitionMessage
# from server.message.image_message import ImageMessage
# from server.message.image_to_text_message import ImageToTextMessage
# from server.message.register_face_recognition_message import RegisterFaceRecognitionMessage
# from server.message.remove_person_message import RemovePersonMessage
# from server.message.start_face_recognition_message import StartFaceRecognitionMessage
# from server.message.vqa_message import VqaMessage
