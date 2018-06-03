import json
import pickle
import socket as st


class ConnectionHelper:
    @staticmethod
    def is_available_port(port_number):
        sock = st.socket(st.AF_INET, st.SOCK_STREAM)
        sock.settimeout(2)  # 2 Second Timeout
        result = sock.connect_ex(('127.0.0.1', port_number))
        return result == 0

    @staticmethod
    def find_available_port():
        max_port_number = 65535
        for i in range(max_port_number):
            if ConnectionHelper.is_available_port(i):
                return i
        return -1

    @staticmethod
    def find_available_ports():
        port1, port2, port3 = ConnectionHelper.find_available_port(), ConnectionHelper.find_available_port(), ConnectionHelper.find_available_port()
        return [port1, port2, port3]

    @staticmethod
    def send_json(socket, data):
        try:
            serialized = json.dumps(data)
        except (TypeError, ValueError) as e:
            raise Exception('You can only send JSON-serializable data')
        # send the length of the serialized data first
        socket.send('%d\n'.encode() % len(serialized))
        # send the serialized data
        socket.sendall(serialized.encode())

    @staticmethod
    def receive_json(socket):
        view = ConnectionHelper.receive(socket)#.decode()
        try:
            deserialized = json.loads(view)
        except (TypeError, ValueError) as e:
            raise Exception('Data received was not in JSON format')
        return deserialized

    @staticmethod
    def send_pickle(socket, object):
        try:
            serialized = pickle.dumps(object)
        except (TypeError, ValueError) as e:
            raise Exception('You can only send JSON-serializable data')
        # send the length of the serialized data first
        socket.send('%d\n'.encode() % len(serialized))
        # send the serialized data
        socket.sendall(serialized)

    @staticmethod
    def receive_pickle(socket):
        view = ConnectionHelper.receive(socket)
        try:
            deserialized = pickle.loads(view)
        except (TypeError, ValueError) as e:
            raise Exception('Data received was not in JSON format')
        return deserialized

    @staticmethod
    def receive(socket):
        # read the length of the data, letter by letter until we reach EOL
        length_str = ''
        char = socket.recv(1).decode()
        if char == '':
            return char
        while char != '\n':
            length_str += char
            char = socket.recv(1).decode()
        total = int(length_str)
        # use a memoryview to receive the data chunk by chunk efficiently
        view = memoryview(bytearray(total))
        next_offset = 0
        while total - next_offset > 0:
            recv_size = socket.recv_into(view[next_offset:], total - next_offset)
            next_offset += recv_size
        view = view.tobytes()
        return view
