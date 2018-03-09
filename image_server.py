#!/usr/bin/env python3

import logging
import struct
from socketserver import StreamRequestHandler, TCPServer
from threading import Thread

import numpy as np

class ImageHandler(StreamRequestHandler):
    def __init__(self, *args, **kwargs):
        StreamRequestHandler.__init__(self, *args, **kwargs)
        print('Connected client {}:{}'.format(*self.client_address))

    def handle(self):
        # self.rfile is a file-like object created by the handler;
        # we can now use e.g. readline() instead of raw recv() calls
        data = self.rfile.readline().strip().decode().split(' ')
        print("{} wrote: {}".format(self.client_address[0], data))
        # Likewise, self.wfile is a file-like object used to write back
        # to the client

        obj = self.server.objects.get(data[1], None)
        if obj is None:
            print('Object "{}" not found'.format(data[1]))
            return self.returnImage(None)

        img = obj.getImage(data[2])
        if img is None:
            print('Image "{}" not found on object {}'.format(data[2], obj))
            return self.returnImage(None)

        self.returnImage(img)

    def returnImage(self, image):
        if image is None:
            image = np.zeros([0,0,0], dtype='uint8')

        if len(image.shape) == 2:
            self.wfile.write(struct.pack('HHH', image.shape[0], image.shape[1], 1))
        elif len(image.shape) == 3:
            self.wfile.write(struct.pack('HHH', image.shape[0], image.shape[1], image.shape[2]))
        self.wfile.write(image.tobytes())

class ImageServer(TCPServer, Thread):
    """docstring for ImageServer"""
    def __init__(self, *, host="0.0.0.0", port=1150, objects={}):
        TCPServer.allow_reuse_address = True
        TCPServer.__init__(self, (host, port), ImageHandler)
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.objects = objects;

    def start(self):
        Thread.start(self)
        return self

    def run(self):
        print("Starting server, listening on {}:{}".format(self.host, self.port))
        self.serve_forever()

if __name__ == "__main__":
    s = ImageServer()
    s.start()
    s.join()
    