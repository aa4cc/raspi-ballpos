import io
import numpy as np
import cv2

class Processor(io.BytesIO):
    def __init__(self, detectors, callback=None):
        super().__init__()
        self.frame_number = 0;
        self.callback = callback
        self.mask = None
        self.mask_dwn = None
        self.detectors = []

        for detector in detectors:
            cls = detector.pop('type')
            self.detectors.append(cls(processor=self, **detector))
        print(self.detectors)

    def write(self, b):
        if params["verbose"] > 0:
            e1 = cv2.getTickCount()

        data = np.fromstring(b, dtype=np.uint8)
        image = np.resize(data,(params["resolution"][1], params["resolution"][0], 3))

        centers = self.processImage(image)

        if self.callback:
            self.callback(center)
        
        if params['verbose']:
            e2 = cv2.getTickCount()
            elapsed_time = (e2 - e1)/ cv2.getTickFrequency()
            print('Frame: {}, center {}, elapsed time: {}'.format(self.frame_number, center, elapsed_time))


    def processImage(self, image):
        return [detector.processImage(self.frame_number, image) for detector in self.detectors]

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_number >= params["num_frames"] > 0:
            raise StopIteration("Number of frames done")
        return self