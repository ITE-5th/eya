import base64
import os
import socket
import threading
from shutil import copy2, rmtree

import cv2
import numpy as np
import time

from face_recognition.face_recognition_model import FaceRecognitionModel
from file_path_manager import FilePathManager
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
        self.image_to_text = ImageToTextModel()
        self.vqa = VqaModel()
        print("finish vqa + image to text")
        self.client_socket, self.address = None, None

    def handle_client_connection(self, client_socket):
        face_recognition = None
        number_of_faces = 10
        try:
            while True:
                message = Helper.receive_json(client_socket)
                print(message)
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
                        path = FilePathManager.resolve("face_recognition/recognition/models")
                        base_model_path = f"{path}/base_model.model"
                        person_path = f"{path}/{name}"
                        if os.path.exists(person_path):
                            rmtree(person_path)
                        os.makedirs(person_path)
                        model_path = f"{person_path}/model.model"
                        copy2(base_model_path, model_path)
                        result["result"] = "success"
                        result["registered"] = True
                    elif type == "start-face-recognition":
                        try:
                            face_recognition = FaceRecognitionModel(name)
                            print("enter face")
                            result["result"] = "success"
                        except FileNotFoundError:
                            result["result"] = "error"
                    elif type == "face-recognition":
                        if face_recognition is not None:
                            result["result"] = face_recognition.predict(image)
                        else:
                            result["result"] = "error"
                    elif type == "add-person":
                        if face_recognition is not None:
                            images = []
                            for i in range(number_of_faces):
                                image, _, _, _ = Server.get_data(message)
                                images.append(image)
                            face_recognition.add_person(name, images)
                            result["result"] = "success"
                        else:
                            result["result"] = "error"
                    elif type == "remove-person":
                        if face_recognition is not None:
                            face_recognition.remove_person(name)
                            result["result"] = "success"
                        else:
                            result["result"] = "error"

                    # Visual Question Answering
                    elif type == "visual-question-answering":
                        result["result"] = self.vqa.predict(question, image)

                    # Image To Text
                    elif type == "image-to-text":
                        result["result"] = self.image_to_text.predict(image)
                    Helper.send_json(client_socket, result)

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
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
