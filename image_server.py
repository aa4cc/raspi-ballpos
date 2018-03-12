#!/usr/bin/env python3

import logging
import struct
from socketserver import StreamRequestHandler, TCPServer
from threading import Thread

import numpy as np
import cv2

METHOD_MAP={
    'GET': "http",
    'RAW': "raw"
}

class ImageHandler(StreamRequestHandler):
    def __init__(self, *args, **kwargs):
        StreamRequestHandler.__init__(self, *args, **kwargs)
        print('Connected client {}:{}'.format(*self.client_address))

    def handle(self):
        # self.rfile is a file-like object created by the handler;
        # we can now use e.g. readline() instead of raw recv() calls
        line = self.rfile.readline()
        print("{} request: {}".format(self.client_address[0], line))
        request = line.strip().decode().split(' ')
        if not len(request):
            return
        method = request[0]
        path = request[1].split('/')
        self.handlePath(path, method)

    def handlePath(self, path, method):
        path.pop(0)
        print("Request path: {}".format(path))
        # Likewise, self.wfile is a file-like object used to write back
        # to the client

        protocol = METHOD_MAP[method]

        obj = self.server.objects.get(path[0], None)
        if obj is None:
            print('Object "{}" not found'.format(path[0]))
            return self.returnImage(None, protocol)

        if len(path) < 2:
            path.append('any')

        img = obj.getImage(path[1])
        if img is None:
            print('Image "{}" not found on object {}'.format(path[1], obj))
            return self.returnImage(None, protocol)

        self.returnImage(img, protocol)

    def returnImage(self, image, protocol="raw"):
        if protocol == "http":
            print("Protocol is HTTP")
            if image is None:
                self.wfile.write(b"HTTP/1.x 404 Image not found\n")
            else:
                self.wfile.write(b"HTTP/1.x 200 OK\n")
            self.wfile.write(b"Cache-Control: no-cache, no-store, must-revalidaten")
            self.wfile.write(b"Pragma: no-cache\n")
            self.wfile.write(b"Expires: 0\n")

            if image is None:
                self.wfile.write(b"Content-Type: text/html; charset=UTF-8\n\n")
                self.wfile.write(b"<html><head><title>Error 404 Image not found</title></head><body><h1>Error 404 Image not found</h1></body></html>\n")
                return

            self.wfile.write(b"Content-Type: image/bmp; charset=UTF-8\n\n")
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _, bmp = cv2.imencode(".bmp", image)
            self.wfile.write(bmp)

        elif protocol == "raw":
            print("Protocol is RAW")
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
    