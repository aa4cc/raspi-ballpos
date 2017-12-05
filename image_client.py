#!/usr/bin/env python3

import socket
import sys

HOST, PORT = "localhost", 1150

# Create a socket (SOCK_STREAM means a TCP socket)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    # Connect to server and send data
    sock.connect((HOST, PORT))
    sock.sendall("My image please!!\n".encode())

    # Receive data from the server and shut down
    file = sock.makefile('rb')
    data = file.read()
    print("Received: {} bytes".format(len(data)))