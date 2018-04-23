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
from misc.connection_helper import ConnectionHelper
from server.message.add_person_message import AddPersonMessage
from server.message.end_add_person_message import EndAddPersonMessage
from server.message.face_recognition_message import FaceRecognitionMessage
from server.message.image_message import ImageMessage
from server.message.image_to_text_message import ImageToTextMessage
from server.message.name_message import NameMessage
from server.message.register_face_recognition_message import RegisterFaceRecognitionMessage
from server.message.remove_person_message import RemovePersonMessage
from server.message.start_face_recognition_message import StartFaceRecognitionMessage
from server.message.vqa_message import VqaMessage
from vqa.vqa_model import VqaModel

# just to use it
Vocabulary()


class LocalServer:
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
        images = []
        base_path = FilePathManager.resolve("saved_images")
        try:
            face_recognition = None
            while True:
                message = ConnectionHelper.receive_pickle(client_socket)
                result = {
                    "result": "success",
                }
                if type == 'close':
                    break
                if isinstance(message, ImageMessage):
                    image = self.to_image(message.image)
                if isinstance(message, NameMessage):
                    name = message.name
                # Face Recognition
                if isinstance(message, RegisterFaceRecognitionMessage):
                    FaceRecognitionModel.register(name, remove_dir=False)
                    result["result"] = "success"
                    result["registered"] = True
                if isinstance(message, StartFaceRecognitionMessage):
                    try:
                        face_recognition = FaceRecognitionModel(name)
                        result["result"] = "success"
                    except FileNotFoundError:
                        result["result"] = "error"
                if isinstance(message, FaceRecognitionMessage):
                    if face_recognition is not None:
                        result["result"] = face_recognition.predict(image)
                    else:
                        result["result"] = "error"
                if isinstance(message, AddPersonMessage):
                    cv2.imwrite(f"{base_path}/image_{len(images) + 1}.jpg", image)
                    images.append(image)
                    result["result"] = "success"
                if isinstance(message, EndAddPersonMessage):
                    if face_recognition is not None:
                        face_recognition.add_person(name, images)
                        images = []
                        result["result"] = "success"
                    else:
                        result["result"] = "error"
                if isinstance(message, RemovePersonMessage):
                    if face_recognition is not None:
                        face_recognition.remove_person(name)
                        result["result"] = "success"
                    else:
                        result["result"] = "error"

                # Visual Question Answering
                if isinstance(message, VqaMessage):
                    result["result"] = self.vqa.predict(message.question, image)

                # Image To Text
                if isinstance(message, ImageToTextMessage):
                    result["result"] = self.image_to_text.predict(image)
                ConnectionHelper.send_json(client_socket, result)
                print("result:")
                print(result)

        finally:
            print('client_socket.close')
            client_socket.close()

    @staticmethod
    def get_data(message):
        image = LocalServer.get(message, "image")
        image = LocalServer.to_image(image) if image is not None else None
        return image, LocalServer.get(message, "question"), LocalServer.get(message, "type"), LocalServer.get(message,
                                                                                                              "name")

    @staticmethod
    def get(message, attr):
        return message[attr] if attr in message else None

    @staticmethod
    def to_image(img_data):
        return cv2.imdecode(np.fromstring(base64.decodebytes(img_data.encode()), np.uint8), cv2.IMREAD_COLOR)

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
    server = LocalServer(port=8888)
    # server = LocalServer(host="192.168.43.71", port=8888)

    try:
        server.start()
    finally:
        server.close()
