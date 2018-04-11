import io
import numpy as np
import cv2
import math
import logging
import warnings
from profilehooks import profile

import sys
sys.path
sys.path.insert(0, './find_object')

import find_object

logger = logging.getLogger(__name__)
NAN = float('nan')

class Detector(object):
    def stop(self):
        pass

    def processImage(self, frame_number, image):
        raise RuntimeError("Class {} as child of detector.Detector must override Detector.proccessImage()".format(self.__class__.__name__))

    def getImage(self, name):
        return None;

class ObjectDetector(Detector):
    def __init__(self, **kwargs):
        self.objectlim = kwargs["object_size"][0] * 255, kwargs["object_size"][1] * 255
        self.color_coefs = kwargs["color_coefs"]
        self.downsample = kwargs["downsample"]
        self.threshold = kwargs["threshold"]
        self.tracking_window = kwargs["tracking_window"]
        self.mask = kwargs.get("mask", None)
        if self.mask:
            self.mask_dwn = self.mask[::self.downsample, ::self.downsample]
        else:
            self.mask_dwn = None
        self.name = kwargs["name"]
        self.debug = kwargs.get("debug", 0)
        self.images = {}
        self.compute_orientation = kwargs.get('compute_orientation', False);
        self.orientation_offset = kwargs.get("orientation_offset", 0)

        print("Object mass must be between {:.0f} px^2 and {:.0f} px^2".format(self.objectlim[0]/255, self.objectlim[1]/255))
        print("Image channel combination coefficients: ({})".format(self.color_coefs))

    def __repr__(self):
        return "<{}(color_coefs={}, threshold={})>".format(self.__class__.__name__, self.color_coefs, self.threshold)

    @property
    def tracking_window_halfsize(self):
        return self.tracking_window//2

    def getImage(self, name):
        return self.images.get(name, None)

    #@profile
    def findTheObject(self, image, object_size_lim = None, mask = None, name = None, compute_orientation = False):
        # Initialize orientation and center to None
        orientation = None
        center = None

        # take a linear combination of the color channels to get a grayscale image containg only images of a desired color
        im = np.clip(image[:,:,0]*self.color_coefs[0] + image[:,:,1]*self.color_coefs[1] + image[:,:,2]*self.color_coefs[2], 0, 255).astype(np.uint8)

        # apply a mask if it is given
        if mask is not None:
            im = cv2.bitwise_and(im, self.im, mask = mask)

        # threshold the image
        im_thrs = cv2.inRange(im, self.threshold,255)

        # Store the image
        self.images[name] = im
        self.images[name+"_thrs"] = im_thrs

        # Compute the moments
        M = cv2.moments(im_thrs)
        # Check whether the area of the found object is within specified limits
        if M['m00'] > 0 and (object_size_lim is None  or object_size_lim[0] < M['m00'] < object_size_lim[1]):
            # Compute the cetner of the object
            center =  M['m10'] / M['m00'], M['m01'] / M['m00']

            # Compute orientation if the corresponding flag is active
            if compute_orientation:
                orientation = self.findOrientation(im_thrs, M, center)

        else:
            if self.debug:
                print("Ball mass out of mass ranges. (mass={}, lim={})".format(M['m00'], object_size_lim))

        return center[0], center[1], orientation

    def findOrientation(self, im_thrs, M, center):
        # Detailed description of this algorithm can be found at http://raphael.candelier.fr/?blog=Image%20Moments

        # Central moments (intermediary step)
        a = M['m20']/M['m00'] - center[0]**2; # mu20/m00
        b = 2*(M['m11']/M['m00'] - center[0]*center[1]); #2*(mu11/m00)
        c = M['m02']/M['m00'] - center[1]**2; # mu02/m00

        # Minor and major axis
        #w = math.sqrt(6*(a+c-math.sqrt(b**2+(a-c)**2)))/2
        #l = math.sqrt(6*(a+c+math.sqrt(b**2+(a-c)**2)))/2

        # Orientation (radians)
        theta = 1/2*math.atan2(b, a-c)

        # Object's pixels coordinates
        j, i = np.nonzero(im_thrs)

        # rot = np.array([-math.sin(theta), math.cos(theta)])
        # coords = np.array([i-center[0], j-center[1]])
        tmp = (i-center[0])*(-math.sin(theta)) + (j-center[1])*math.cos(theta)

        if np.sum(tmp**3) > 0:
            # Fix direction
            theta += math.pi;

        theta += self.orientation_offset

        return np.mod(theta,2*math.pi)

    def processImage(self, frame_number, image):
        try:
            start_y = 0
            end_y = params["resolution"][1]
            start_x = 0
            end_x = params["resolution"][0]
            # Downsample the image
            if self.downsample > 1:
                image_dwn = image[::self.downsample, ::self.downsample, :]
                object_lim_dwn = self.objectlim[0]//(self.downsample**2), self.objectlim[1]//(self.downsample**2)

                location = self.findTheObject(image_dwn, object_size_lim=object_lim_dwn, mask = self.mask_dwn, name="whole")
                if not location:
                    if self.debug:
                        print('The object was not found in the whole image!')
                    return None

                center = int(self.downsample*location[0]), int(self.downsample*location[1])

                halfsize = self.tracking_window_halfsize
                # Find the ball in smaller image
                start_y = max((center[1]-halfsize), 0)
                end_y = min((center[1]+halfsize), params["resolution"][1])
                start_x = max((center[0]-halfsize), 0)
                end_x = min((center[0]+halfsize), params["resolution"][0])

                self.images["image_dwn"] = image_dwn

            self.images["image"] = image

            imageROI = image[start_y:end_y, start_x:end_x, :]

            self.images["imageROI"] = imageROI

            if self.mask is not None:
                mask_ROI = self.mask[start_y:end_y, start_x:end_x]
            else:
                mask_ROI = None

            # Find the ball in the region of interest
            location_inROI = self.findTheObject(imageROI, object_size_lim=self.objectlim, mask=mask_ROI, name="roi", compute_orientation = self.compute_orientation)
            # If the ball is not found, raise an exception
            if not location_inROI:
                if self.debug:
                    print('The object was not found in the ROI!')
                return None
            
            # transform the measured position from ROI to full image coordinates
            x = start_x + location_inROI[0]
            y = start_y + location_inROI[1]
            if location_inROI[2]:
                theta = location_inROI[2]
            else:
                theta = NAN;
            return x, y, theta
        except Exception as e:
            logger.exception(e)

class BallDetector(ObjectDetector):
    def __init__(self, **kwargs):
        kwargs["object_size"] = (kwargs["ball_size"][0]/2)**2 * math.pi, (kwargs["ball_size"][1]/2)**2 * math.pi
        kwargs["compute_orientation"] = False
        kwargs["orientation_offset"] = 0
        ObjectDetector.__init__(self, **kwargs)

class ObjectDetectorInC(ObjectDetector):
    def processImage(self, frame_number, image):
        im_thrs = np.zeros(image.shape[0:2], 'uint8');
        location = find_object.process_image(
            image=image,
            image_thrs=im_thrs,
            color_coefs=self.color_coefs,
            threshold=self.threshold,
            compute_orientation=self.compute_orientation,
            downsample=self.downsample,
            window_size=self.tracking_window,
            orientation_offset=self.orientation_offset
            )
        self.images["thrs"] = im_thrs
        return location

    def getImage(self, name):
        im = self.images.get(name, None);
        if im is None:
            im = self.images.get("thrs", None);
        return im