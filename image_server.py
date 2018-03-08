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
        data = self.rfile.readline().strip()
        print("{} wrote: {}".format(self.client_address[0], data.decode()))
        # Likewise, self.wfile is a file-like object used to write back
        # to the client
        image = self.server.image;
        if len(image.shape) == 2:
            self.wfile.write(struct.pack('HHH', image.shape[0], image.shape[1], 1))
        elif len(image.shape) == 3:
            self.wfile.write(struct.pack('HHH', image.shape[0], image.shape[1], image.shape[2]))
        self.wfile.write(image.tobytes())

class ImageServer(TCPServer, Thread):
    """docstring for ImageServer"""
    def __init__(self, *, host="0.0.0.0", port=1150):
        TCPServer.allow_reuse_address = True
        TCPServer.__init__(self, (host, port), ImageHandler)
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.image = np.zeros([480,480,3], dtype='uint8')

    def start(self):
        Thread.start(self)
        return self

    def run(self):
        print("Starting server, listening on {}:{}".format(self.host, self.port))
        self.serve_forever()
        
    def writeImage(self, im):
        self.image = im

if __name__ == "__main__":
    s = ImageServer()
    s.start()
    s.join()
    