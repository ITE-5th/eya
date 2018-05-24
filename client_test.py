import base64
import socket

from file_path_manager import FilePathManager
from misc.connection_helper import ConnectionHelper
from server.message.face_recognition_message import FaceRecognitionMessage
from server.message.image_to_text_message import ImageToTextMessage
from server.message.register_face_recognition_message import RegisterFaceRecognitionMessage
from server.message.start_face_recognition_message import StartFaceRecognitionMessage
from server.message.vqa_message import VqaMessage


class Client:
    def __init__(self, host=socket.gethostname(), port=8888):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):
        print('connected to server ' + self.host + ':' + str(self.port))
        self.socket.connect((self.host, self.port))
        name = "zaher"
        self.communicate_with_server(RegisterFaceRecognitionMessage(name))
        self.communicate_with_server(StartFaceRecognitionMessage(name))
        i = 0
        while True:
            if i % 3 == 0:
                message = self._build_message("face-recognition")
            elif i % 3 == 1:
                message = self._build_message("image-to-text")
            else:
                message = self._build_message("visual-question-answering", "what is the color of the door?")
            self.communicate_with_server(message)
            i += 1

    def close(self):
        self.socket.close()

    def communicate_with_server(self, message):
        ConnectionHelper.send_pickle(self.socket, message)
        response = ConnectionHelper.receive_json(self.socket)
        print(response)

    @staticmethod
    def _build_message(type, question=None):
        if type == "face-recognition":
            file_path = FilePathManager.resolve("face/test_faces/20.jpg")
        else:
            file_path = FilePathManager.resolve("vqa/test_images/test.jpg")
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        if type == "face-recognition":
            message = FaceRecognitionMessage(encoded_string)
        elif type == "visual-question-answering":
            message = VqaMessage(encoded_string, question)
        else:
            message = ImageToTextMessage(encoded_string)
        return message


if __name__ == '__main__':
    client = Client(port=9500)
    # client = Client(host="192.168.43.71", port=8888)
    client.start()
