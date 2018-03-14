import base64
import os
import socket
import threading
import numpy as np
import cv2

from face_recognition.face_recognition_model import FaceRecognitionModel
from helper import Helper
from image_to_text.build_vocab import Vocabulary
from image_to_text.image_to_text_model import ImageToTextModel
from vqa.vqa_model import VqaModel

# just to use it
Vocabulary()


class Server:
    def __init__(self, host=socket.gethostname(), port=8888):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen(5)
        self.face_recognition = FaceRecognitionModel()
        self.image_to_text = ImageToTextModel()
        self.vqa = VqaModel()
        self.client_socket, self.address = None, None

    def handle_client_connection(self, client_socket):
        while True:
            message = Helper.receive_json(client_socket)
            if message != '':
                img_data, question, type = self.get_data(message)
                nparr = np.fromstring(base64.decodebytes(img_data.encode()), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                result = {"result": question}
                if type == "visual-question-answering":
                    result["result"] = self.vqa.predict(question, image)
                elif type == "face-recognition":
                    result["result"] = self.face_recognition.predict(image)
                elif type == "image-to-text":
                    result["result"] = self.image_to_text.predict(image)
                Helper.send_json(client_socket, result)

    def get_data(self, message):
        type = ''
        img_data = ''
        question = ''
        try:
            type = message['type'].lower()
            img_data = message["image"]
            question = message["question"]
        finally:
            return img_data, question, type

    def start(self):
        print('server started at {}:{}'.format(self.host, str(self.port)))
        while True:
            client_socket, address = self.socket.accept()
            print('Accepted connection from {}:{}'.format(address[0], address[1]))

            client_handler = threading.Thread(
                target=self.handle_client_connection,
                args=(client_socket,)
            )
            client_handler.start()

    def close(self):
        self.socket.close()


if __name__ == '__main__':
    os.system('ps -fA | grep python | tail -n1 | awk \'{ print $3 }\'|xargs kill')
    server = Server()

    try:
        server.start()
    finally:
        server.close()
