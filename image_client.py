#!/usr/bin/env python3

import socket
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
import click

@click.command()
@click.option("--host", "-h", default="147.32.86.182", help="Address of image server")
@click.option("--port", "-p", default=1150, help="Port of image server")
@click.argument("object", default="Processor")
@click.argument("name")
def main(host, port, object, name):
	# Create a socket (SOCK_STREAM means a TCP socket)
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
	    # Connect to server and send data
	    sock.connect((host, port))
	    sock.sendall("GET {} {}\n".format(object, name).encode())

	    # Receive data from the server and shut down
	    file = sock.makefile('rb')
	    res = struct.unpack('HHH', file.read(6))
	    if res[2] == 1:
	    	res = res[0:2]
	    data = np.reshape(np.frombuffer(file.read(), dtype='uint8'), res)

	    print(data.shape)

	    plt.imshow(data)
	    plt.show()

if __name__ == '__main__':
	main()