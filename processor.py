import io
import numpy as np
import cv2
from profilehooks import profile
import image_server
from pprint import pprint
from multiprocessing import Process, Event, Queue, Condition
from sharemem import SharedMemory
import math

class Processor(io.BytesIO):
    def __init__(self, detectors, callback=None, mask=None):
        super().__init__()
        self.frame_number = 0;
        self.callback = callback
        self.detectors = []
        self.image_server = None

        for detector in detectors:
            cls = detector.pop('type')
            self.detectors.append(cls(mask=mask, **detector))

        if self.detectors:
            print("Active detectors:")
            for d in self.detectors:
                print(" {}:\t{}".format(d.name,  d))
        else:
            print('No detectors active')

        if params['image_server']:
            print("Starting TCP image server")
            objects = {}
            for detector in self.detectors:
                objects["Detector-{}".format(detector.name)] = detector
            objects["Processor"] = self

            pprint(objects)

            self.image_server = image_server.ImageServer(objects=objects)
            self.image_server.start()

    def getImage(self, name):
        return self.image

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_number >= params["num_frames"] > 0:
            raise StopIteration("Number of frames done")
        return self

    def __repr__(self):
        return "<{}.{}(len(detectors)={})>".format(self.__module__, self.__class__.__name__, len(self.detectors))

    def stop(self):
        for detector in self.detectors:
            detector.stop()

        if self.image_server:
            self.image_server.shutdown()
        
class SingleCore(Processor):
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


            c = ", ".join(("({0:6.2f}, {1:6.2f})" if math.isnan(center[2]) else "({0:6.2f}, {1:6.2f}, {2:6.4f})").format(*center) if center else "None" for center in centers)

            print('Frame: {:5}, center [{}], elapsed time: {:.1f}ms'.format(self.frame_number, c, elapsed_time*1000))

        self.frame_number += 1;


    def processImage(self, image):
        return [detector.processImage(self.frame_number, image) for detector in self.detectors]

class MultiCore(Processor):
    """docstring for MultiCoreDetector"""
    def __init__(self, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        key = kwargs.get("key", 123456)
        self.sm = SharedMemory(key, params["resolution"][0]*params["resolution"][0]*3)
        self.start_cond = Condition()
        self.stop_event = Event()
        self.workers = [MultiCore.Worker(detector=detector, key = key, start_cond=self.start_cond, stop_event=self.stop_event).start() for detector in self.detectors]
        print("Multicore workers:")
        for worker in self.workers:
            print("  Woker: {}, pid={}".format(worker.name, worker.pid))

    def stop(self):
        Processor.stop(self)
        self.stop_event.set()
        for worker in self.workers:
            worker.join()

    def write(self, b):
        if params["verbose"] > 0:
            e1 = cv2.getTickCount()

        centers = self.processImage(b)

        if self.callback:
            self.callback(centers)
        
        if params['verbose']:
            e2 = cv2.getTickCount()
            elapsed_time = (e2 - e1)/ cv2.getTickFrequency()


            c = ", ".join(("({0:6.2f}, {1:6.2f})" if math.isnan(center[2]) else "({0:6.2f}, {1:6.2f}, {2:6.4f})").format(*center) if center else "None" for center in centers)

            print('Frame: {:5}, center [{}], elapsed time: {:.1f}ms'.format(self.frame_number, c, elapsed_time*1000))

        self.frame_number += 1;

        if self.stop_event.is_set():
            raise StopIteration("Number of frames done")

        self.sm.write(image)
        with self.start_cond:
            self.start_cond.notify_all()
        return [worker.get_result(timeout=0.5) for worker in self.workers]

    class Worker(Process):
        """docstring for Worker"""
        def __init__(self, detector, start_cond, stop_event, key):
            Process.__init__(self, name=detector.name, daemon=True)
            self.detector = detector
            self.start_cond = start_cond
            self.stop_event = stop_event
            self.result_queue = Queue(1)
            self.key = key
        def start(self):
            Process.start(self)
            return self

        def run(self):
            self.sm = SharedMemory(self.key, params["resolution"][0]*params["resolution"][0]*3, create=False)
            try:
                while not self.stop_event.is_set():
                    with self.start_cond:
                        c = self.start_cond.wait(0.5)
                    if c:
                        if params["verbose"] > 1:
                            e1 = cv2.getTickCount()
                        data = np.fromstring(self.sm.read(lock=False), dtype=np.uint8)
                        image = np.resize(data,(params["resolution"][1], params["resolution"][0], 3))
                        center = self.detector.processImage(-1, image)
                        self.result_queue.put(center, timeout=0.5)
                        if params["verbose"] > 1:
                            e2 = cv2.getTickCount()
                            elapsed_time = (e2 - e1)/ cv2.getTickFrequency()
                            print("{}: {:.2f}ms".format(self.name, elapsed_time*1000))
            except KeyboardInterrupt:
                self.stop_event.set()

        def get_result(self, *args, **kwargs):
            try:
                return self.result_queue.get(*args, **kwargs)
            except Queue.Empty:
                print("Lost sample in {} process".format(self.name))
                return None

