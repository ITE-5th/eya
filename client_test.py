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
        i = 1
        while True:
            i += 1
            self.communicate_with_server(
                self._build_message("image-to-text" if i % 2 == 0 else "visual-question-answering",
                                    "what is the color of the umbrella?"))

    def close(self):
        self.socket.close()

    def communicate_with_server(self, message):
        Helper.send_json(self.socket, message)
        response = Helper.receive_json(self.socket)
        print(response)

    def _build_message(self, type, question=None):
        file_path = FilePathManager.resolve("vqa/test_images/girl_with_umbrella.jpg")
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        json_data = {"type": type, "image": encoded_string, "question": question}
        return json_data


if __name__ == '__main__':
    client = Client()
    client.start()
