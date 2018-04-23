import cv2
from multipledispatch import dispatch

from face.face_recognition_model import FaceRecognitionModel
from file_path_manager import FilePathManager
from misc.connection_helper import ConnectionHelper
from server.message.add_person_message import AddPersonMessage
from server.message.close_message import CloseMessage
from server.message.end_add_person_message import EndAddPersonMessage
from server.message.face_recognition_message import FaceRecognitionMessage
from server.message.image_message import ImageMessage
from server.message.image_to_text_message import ImageToTextMessage
from server.message.register_face_recognition_message import RegisterFaceRecognitionMessage
from server.message.remove_person_message import RemovePersonMessage
from server.message.start_face_recognition_message import StartFaceRecognitionMessage
from server.message.vqa_message import VqaMessage


class RequestHandler:
    def __init__(self, vqa, itt):
        self.vqa = vqa
        self.itt = itt
        self.face_recognition = None
        self.images = []
        self.base_path = FilePathManager.resolve("saved_images")

    @dispatch(RegisterFaceRecognitionMessage)
    def handle_message(self, message):
        result = {

        }
        FaceRecognitionModel.register(message.name, remove_dir=False)
        result["result"] = "success"
        result["registered"] = True
        return result

    @dispatch(StartFaceRecognitionMessage)
    def handle_message(self, message):
        result = {

        }
        try:
            self.face_recognition = FaceRecognitionModel(message.name)
            result["result"] = "success"
        except FileNotFoundError:
            result["result"] = "error"
        return result

    @dispatch(AddPersonMessage)
    def handle_message(self, message):
        result = {

        }
        cv2.imwrite(f"{self.base_path}/image_{len(self.images) + 1}.jpg", message.image)
        self.images.append(message.image)
        result["result"] = "success"
        return result

    @dispatch(EndAddPersonMessage)
    def handle_message(self, message):
        result = {

        }
        if self.face_recognition is not None:
            self.face_recognition.add_person(message.name, self.images)
            self.images = []
            result["result"] = "success"
        else:
            result["result"] = "error"
        return result

    @dispatch(RemovePersonMessage)
    def handle_message(self, message):
        result = {

        }
        if self.face_recognition is not None:
            self.face_recognition.remove_person(message.name)
            result["result"] = "success"
        else:
            result["result"] = "error"
        return result

    @dispatch(FaceRecognitionMessage)
    def handle_message(self, message):
        result = {

        }
        if self.face_recognition is not None:
            result["result"] = self.face_recognition.predict(message.image)
        else:
            result["result"] = "error"
        return result

    @dispatch(VqaMessage)
    def handle_message(self, message):
        result = {
            "result": self.vqa.predict(message.question, message.image)
        }
        return result

    @dispatch(ImageToTextMessage)
    def handle_message(self, message):
        result = {
            "result": self.itt.predict(message.image)
        }
        return result

    def start(self, client_socket):
        try:
            while True:
                message = ConnectionHelper.receive_pickle(client_socket)
                if isinstance(message, CloseMessage):
                    break
                if isinstance(message, ImageMessage):
                    message.image = ConnectionHelper.to_image(message.image)
                result = self.handle_message(message)
                ConnectionHelper.send_json(client_socket, result)
                print("result:")
                print(result)
        finally:
            print("socket closed")
            client_socket.close()
