import io
import numpy as np
import cv2
from profilehooks import profile
import matplotlib.pyplot as plt
import image_server

class Processor(io.BytesIO):
    def __init__(self, detectors, callback=None):
        super().__init__()
        self.frame_number = 0;
        self.callback = callback
        self.mask = None
        self.mask_dwn = None
        self.detectors = []
        self.image_server = None

        for detector in detectors:
            cls = detector.pop('type')
            self.detectors.append(cls(processor=self, **detector))

        if self.detectors:
            print("Active detectors:")
            for d in self.detectors:
                print(" {}:\t{}".format(d.name,  d))
        else:
            print('No detectors active')

        if params['image_server']:
            print("Starting TCP image server")
            self.image_server = image_server.ImageServer()
            self.image_server.start()

    #@profile
    def write(self, b):
        if params["verbose"] > 0:
            e1 = cv2.getTickCount()

        data = np.fromstring(b, dtype=np.uint8)
        self.image = np.resize(data,(params["resolution"][1], params["resolution"][0], 3))

        centers = self.processImage(self.image)

        if self.callback:
            self.callback(centers)
        
        if params['verbose']:
            e2 = cv2.getTickCount()
            elapsed_time = (e2 - e1)/ cv2.getTickFrequency()

            c = ", ".join("({x:6.2f}, {y:6.2f})".format(x=center[0], y=center[1]) if center else "None" for center in centers)

            print('Frame: {}, center [{}], elapsed time: {:.1f}ms'.format(self.frame_number, c, elapsed_time*1000))

        if self.image_server:
            self.image_server.writeImage(self.image)

        # if self.frame_number > 30:
        #     cv2.imwrite("img.png",image)
        #     import sys
        #     sys.exit(0)

        self.frame_number += 1;

    def stop(self):
        for detector in self.detectors:
            detector.stop()

        if self.image_server:
            self.image_server.shutdown()


    def processImage(self, image):
        return [detector.processImage(self.frame_number, image) for detector in self.detectors]

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_number >= params["num_frames"] > 0:
            raise StopIteration("Number of frames done")
        return self