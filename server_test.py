import base64
import os
import socket
import threading

# import cv2
import numpy as np

# from face_recognition.face_recognition_model import FaceRecognitionModel
# just to use it
# Vocabulary()
from misc.json_helper import JsonHelper


# from image_to_text.build_vocab import Vocabulary
# from image_to_text.image_to_text_model import ImageToTextModel
# from vqa.vqa_model import VqaModel


class Server:
    def __init__(self, host=socket.gethostname(), port=8888):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen(5)
        # self.image_to_text = ImageToTextModel()
        # self.vqa = VqaModel()
        print("finish vqa + image to text")
        self.client_socket, self.address = None, None

    def handle_client_connection(self, client_socket):
        face_recognition = None
        number_of_faces = 10

        try:
            while True:
                message = JsonHelper.receive_json(client_socket)
                # print(message)
                if message != '':
                    image, question, type, name = Server.get_data(message)
                    if name is not None:
                        name = name.lower().replace(" ", "_")
                    result = {
                        "result": "error",
                    }
                    if type == 'close':
                        break

                    # Face Recognition
                    elif type == "register-face-recognition":
                        result["result"] = "register-face-recognition"
                        result["registered"] = True
                    elif type == "start-face-recognition":
                        result["result"] = "start-face-recognition"
                    elif type == "face-recognition":
                        result["result"] = "face-recognition"
                    elif type == "add-person":
                        result["result"] = "add-person"
                    elif type == "remove-person":
                        result["result"] = "remove-person"
                    # Visual Question Answering
                    elif type == "visual-question-answering":
                        result["result"] = 'visual-question-answering'
                    # Image To Text
                    elif type == "image-to-text":
                        result["result"] = 'image-to-text'
                    JsonHelper.send_json(client_socket, result)

        finally:
            print('client_socket.close')
            client_socket.close()

    @staticmethod
    def get_data(message):
        image = Server.get(message, "image")
        return Server.to_image(image) if image is not None else None, \
               Server.get(message, "question"), \
               Server.get(message, "type"), \
               Server.get(message, "name")

    @staticmethod
    def get(message, attr):
        return message[attr] if attr in message else None

    @staticmethod
    def to_image(img_data):
        nparr = np.fromstring(base64.decodebytes(img_data.encode()), np.uint8)
        return img_data

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
    server = Server(host='192.168.1.4')

    try:
        server.start()
    finally:
        server.close()
