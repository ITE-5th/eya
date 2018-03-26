import base64
import socket

from file_path_manager import FilePathManager
from helper import Helper


class Client:
    def __init__(self, host=socket.gethostname(), port=8888):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):
        print('connected to server ' + self.host + ':' + str(self.port))
        self.socket.connect((self.host, self.port))
        json_data = {
            "type": "register-face-recognition",
            "name": "Obada Jabassini"
        }
        self.communicate_with_server(json_data)
        json_data = {
            "type": "start-face-recognition",
            "name": "Obada Jabassini"
        }
        self.communicate_with_server(json_data)
        i = 0
        while True:
            if i % 3 == 0:
                message = self._build_message("image-to-text")
            elif i % 3 == 1:
                message = self._build_message("visual-question-answering", "what is the color of the door?")
            else:
                message = self._build_message("face-recognition")
            self.communicate_with_server(message)
            i += 1

    def close(self):
        self.socket.close()

    def communicate_with_server(self, message):
        Helper.send_json(self.socket, message)
        response = Helper.receive_json(self.socket)
        print(response)

    @staticmethod
    def _build_message(type, question=None):
        if type == "face-recognition":
            # file_path = FilePathManager.resolve("vqa/test_images/test.jpg")
            file_path = FilePathManager.resolve("face_recognition/test_images/zaher_2.jpg")
        else:
            file_path = FilePathManager.resolve("vqa/test_images/test.jpg")
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        json_data = {"type": type, "image": encoded_string, "question": question}
        return json_data


if __name__ == '__main__':
    client = Client()
    client.start()
