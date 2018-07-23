import sys
import warnings

import cv2
import matplotlib.pyplot as plt
from multipledispatch import dispatch

import server
from face_recognition_model import FaceRecognitionModel
from file_path_manager import FilePathManager
from misc.converter import Converter
from misc.receiver import Receiver
from misc.sender import Sender
from server.message.add_person_message import AddPersonMessage
from server.message.close_message import CloseMessage
from server.message.end_add_person_message import EndAddPersonMessage
from server.message.face_recognition_message import FaceRecognitionMessage
from server.message.image_message import ImageMessage
from server.message.image_to_text_message import ImageToTextMessage
from server.message.object_recognition_message import ObjectRecognitionMessage
from server.message.remove_person_message import RemovePersonMessage
from server.message.start_face_recognition_message import StartFaceRecognitionMessage
from server.message.vqa_message import VqaMessage

warnings.filterwarnings("ignore")

sys.modules['skill-socket_ITE-5th.code'] = server
sys.modules['skill-socket_ITE-5th'] = server


class SocketRequestHandler:
    def __init__(self, vqa, itt, obj_rec):
        self.vqa = vqa
        self.itt = itt
        self.obj_rec = obj_rec
        self.face_recognition = None
        self.images = []
        self.base_path = FilePathManager.resolve("saved_images")

    def register(self, message):
        if not self.face_recognition:
            self.face_recognition = FaceRecognitionModel.create_user(message.user_name)
        return True

    @dispatch(StartFaceRecognitionMessage)
    def handle_message(self, message):
        if not self.register(message):
            return {
                "result": "error"
            }

        return {
            "result": "success"
        }

    @dispatch(AddPersonMessage)
    def handle_message(self, message):
        cv2.imwrite(f"{self.base_path}/image_{len(self.images) + 1}.jpg", message.image)
        self.images.append(message.image)
        return {
            "result": "success"
        }

    @dispatch(EndAddPersonMessage)
    def handle_message(self, message):
        if not self.register(message):
            return {
                "result": "error"
            }
        if self.face_recognition is not None:
            self.face_recognition.add_person(message.name, self.images)
            self.images = []
        return {
            "result": "success" if self.face_recognition is not None else "error"
        }

    @dispatch(RemovePersonMessage)
    def handle_message(self, message):
        if not self.register(message):
            return {
                "result": "error"
            }
        if self.face_recognition is not None:
            self.face_recognition.remove_person(message.name)
        return {
            "result": "success" if self.face_recognition is not None else "error"
        }

    @dispatch(FaceRecognitionMessage)
    def handle_message(self, message):
        if not self.register(message):
            return {
                "result": "error"
            }
        return {
            "result": self.face_recognition.predict(message.image) if self.face_recognition is not None else "error"
        }

    @dispatch(VqaMessage)
    def handle_message(self, message):
        return {
            "result": self.vqa.predict(message.question, message.image)
        }

    @dispatch(ImageToTextMessage)
    def handle_message(self, message):
        return {
            "result": self.itt.predict(message.image)
        }

    @dispatch(ObjectRecognitionMessage)
    def handle_message(self, message):
        return {
            "result": self.obj_rec.predict(message.image, message.object_name)
        }

    @staticmethod
    def show_image(image):
        cv2.imwrite(FilePathManager.resolve('captured.jpg'), image)
        plot_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.cla()
        plt.axis("off")
        plt.imshow(plot_image)
        plt.show()

    def start(self, client_socket):
        try:
            sender = Sender(client_socket, True)
            receiver = Receiver(client_socket, True)
            while True:
                message = receiver.receive()
                message = Converter.to_object(message, json=True)
                if isinstance(message, CloseMessage):
                    break
                if isinstance(message, ImageMessage):
                    message.image = Converter.to_image(message.image)
                    self.show_image(message.image)
                try:
                    result = self.handle_message(message)
                except Exception as e:
                    print(e)
                    result = {
                        "result": "error"
                    }
                sender.send(result)
                print(f"output: {result['result']}")
        except:
            print("socket closed")
        finally:
            client_socket.close()
