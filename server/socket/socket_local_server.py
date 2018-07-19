import os
import socket
import threading

from server.socket.socket_request_handler import SocketRequestHandler
from image_to_text_model import ImageToTextModel
from misc.connection import Connection
from object_recognition_model import ObjectRecognitionModel
from vqa_model import VqaModel



class SocketLocalServer:
    def __init__(self, host=socket.gethostname(), ports=None):
        self.host = host
        if ports is None:
            ports = Connection.find_available_ports()
        self.vqa_port, self.image_to_text_port, self.object_recognition_port, self.face_recognition_port = ports
        self.vqa_socket = self.create_socket(host, self.vqa_port)
        self.image_to_text_socket = self.create_socket(host, self.image_to_text_port)
        self.object_recognition_socket = self.create_socket(host, self.object_recognition_port)
        self.face_recognition_socket = self.create_socket(host, self.face_recognition_port)
        self.vqa = VqaModel()
        self.image_to_text = ImageToTextModel()
        self.object_recognition = ObjectRecognitionModel()

    @staticmethod
    def create_socket(host, port):
        st = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        st.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        st.bind((host, port))
        st.listen(5)
        return st

    def handle_client_connection(self, client_socket):
        handler = SocketRequestHandler(self.vqa, self.image_to_text, self.object_recognition)
        handler.start(client_socket)

    def handle_socket(self, sock):
        while True:
            client_socket, address = sock.accept()
            print('Accepted connection from {}:{}'.format(address[0], address[1]))
            client_handler = threading.Thread(
                target=self.handle_client_connection,
                args=(client_socket,)
            )
            client_handler.start()

    def start(self):
        print(
            f"vqa port = {self.vqa_port}, itt port = {self.image_to_text_port}, object recognition port = {self.object_recognition_port}, face recognition port = {self.face_recognition_port}")
        vqa_thread = threading.Thread(target=self.handle_socket, args=(self.vqa_socket,))
        image_to_text_thread = threading.Thread(target=self.handle_socket, args=(self.image_to_text_socket,))
        object_recognition_thread = threading.Thread(target=self.handle_socket, args=(self.object_recognition_socket,))
        face_recognition_thread = threading.Thread(target=self.handle_socket, args=(self.face_recognition_socket,))
        vqa_thread.start()
        image_to_text_thread.start()
        face_recognition_thread.start()
        object_recognition_thread.start()
        vqa_thread.join()
        image_to_text_thread.join()
        object_recognition_thread.join()
        face_recognition_thread.join()

    def close(self):
        self.vqa_socket.close()
        self.image_to_text_socket.close()
        self.face_recognition_socket.close()
        self.object_recognition_socket.close()


if __name__ == '__main__':
    os.system('ps -fA | grep python | tail -n1 | awk \'{ print $3 }\'| xargs kill')
    number_of_services = 4
    first_port = 9500
    ports = [first_port + i for i in range(number_of_services)]
    server = SocketLocalServer(host=socket.gethostname(), ports=ports)
    # server = SocketLocalServer(host="192.168.43.71", ports=ports)
    try:
        server.start()
    finally:
        server.close()
