import base64
import os
import socket
import threading

import cv2
import numpy as np

from encoder_decoder.build_vocab import Vocabulary
from face.face_recognition_model import FaceRecognitionModel
from file_path_manager import FilePathManager
from image_to_text.image_to_text_model import ImageToTextModel
from misc.json_helper import JsonHelper
from vqa.vqa_model import VqaModel

# just to use it
Vocabulary()


class Server:
    def __init__(self, host=socket.gethostname(), port=9000):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen(5)
        self.image_to_text = ImageToTextModel()
        self.vqa = VqaModel()
        self.client_socket, self.address = None, None

    def handle_client_connection(self, client_socket):
        face_recognition = None
        try:
            images = []
            while True:
                message = JsonHelper.receive_json(client_socket)
                print("message:")
                print(message)
                base_path = FilePathManager.resolve("saved_images")
                if message != '':
                    image, question, type, name = Server.get_data(message)
                    if name is not None:
                        name = name.lower().replace(" ", "_")
                    result = {
                        "result": "success",
                    }
                    if type == 'close':
                        break

                    # Face Recognition
                    elif type == "register-face-recognition":
                        FaceRecognitionModel.register(name, remove_dir=False)
                        result["result"] = "success"
                        result["registered"] = True
                    elif type == "start-face-recognition":
                        try:
                            face_recognition = FaceRecognitionModel(name)
                            result["result"] = "success"
                        except FileNotFoundError:
                            result["result"] = "error"
                    elif type == "face-recognition":
                        if face_recognition is not None:
                            result["result"] = face_recognition.predict(image)
                        else:
                            result["result"] = "error"
                    elif type == "add-person":
                        cv2.imwrite(f"{base_path}/image_{len(images)}.jpg", image)
                        images.append(image)
                        result["result"] = "success"
                    elif type == "end-add-person":
                        if face_recognition is not None:
                            face_recognition.add_person(name, images)
                            images = []
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
                    if type != "add-person":
                        JsonHelper.send_json(client_socket, result)
                    print("result:")
                    print(result)

        finally:
            print('client_socket.close')
            client_socket.close()

    @staticmethod
    def get_data(message):
        image = Server.get(message, "image")
        image = Server.to_image(image) if image is not None else None
        return image, Server.get(message, "question"), Server.get(message, "type"), Server.get(message, "name")

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
    # server = Server(port=8888)
    server = Server(host="192.168.1.7", port=8888)

    try:
        server.start()
    finally:
        server.close()
