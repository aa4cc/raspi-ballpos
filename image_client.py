#!/usr/bin/env python3

import socket
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt

HOST, PORT = "147.32.86.182", 1150

# Create a socket (SOCK_STREAM means a TCP socket)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    # Connect to server and send data
    sock.connect((HOST, PORT))
    sock.sendall("My image please!!\n".encode())

    # Receive data from the server and shut down
    file = sock.makefile('rb')
    res = struct.unpack('HH', file.read(4))
    data = np.reshape(np.frombuffer(file.read(), dtype='uint8'), res +(3,))

    plt.imshow(data)
    plt.show()