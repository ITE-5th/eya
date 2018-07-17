import socket as st


class Connection:
    @staticmethod
    def is_available_port(port_number):
        sock = st.socket(st.AF_INET, st.SOCK_STREAM)
        sock.settimeout(2)  # 2 Second Timeout
        result = sock.connect_ex(('127.0.0.1', port_number))
        return result == 0

    @staticmethod
    def find_available_port(expected):
        max_port_number = 65535
        for i in range(max_port_number):
            if Connection.is_available_port(i) and i not in expected:
                return i
        return -1

    @staticmethod
    def find_available_ports(number=3):
        ports = []
        for i in range(number):
            ports.append(Connection.find_available_port(ports))
        return ports
