import io
import numpy as np
import cv2
import math
import logging
from profilehooks import profile

logger = logging.getLogger(__name__)
NAN = float('nan')

class Detector(object):
    def stop(self):
        pass

    def processImage(self, frame_number, image):
        raise RuntimeError("Class {} as child of detector. Detector must override Detector.proccessImage()")

    def getImage(self, name):
        return None;


class BallDetector(Detector):
    def __init__(self, **kwargs):
        self.ballmasslim = (kwargs["ball_size"][0]/2)**2 * math.pi * 255, (kwargs["ball_size"][1]/2)**2 * math.pi * 255
        self.color_coefs = kwargs["color_coefs"]
        self.downsample = kwargs["downsample"]
        self.threshold = kwargs["threshold"]
        self.tracking_window = kwargs["tracking_window"]
        self.tracking_window_halfsize = self.tracking_window//2
        self.mask = kwargs.get("mask", None)
        if self.mask:
            self.mask_dwn = self.mask[::self.downsample, ::self.downsample]
        else:
            self.mask_dwn = None
        self.name = kwargs["name"]
        self.debug = kwargs.get("debug", 0)
        self.images = {}

        print("Ball mass must be between {:.0f} px^2 and {:.0f} px^2".format(self.ballmasslim[0]/255, self.ballmasslim[1]/255))
        print("Image channel combination coefficients: ({})".format(self.color_coefs))

    def __repr__(self):
        return "<{}(color_coefs={}, threshold={})>".format(self.__class__.__name__, self.color_coefs, self.threshold)

    def getImage(self, name):
        return self.images.get(name, None)

    def findTheObject(self, image, ball_size_lim = None, mask = None, denoise = True, kernel = None, iterations = 2, name=None):
        # take a linear combination of the color channels to get a grayscale image containg only images of a desired color
        im = np.clip(image[:,:,0]*self.color_coefs[0] + image[:,:,1]*self.color_coefs[1] + image[:,:,2]*self.color_coefs[2], 0, 255).astype(np.uint8)

        # apply a mask if it is given
        if mask is not None:
            im = cv2.bitwise_and(im, self.im, mask = mask)

        # threshold the image
        im_thrs = cv2.inRange(im, self.threshold,255)

        # Store the image
        self.images[name] = im

        M = cv2.moments(im_thrs)
        if M['m00'] > 0 and (ball_size_lim is None  or ball_size_lim[0] < M['m00'] < ball_size_lim[1]):
            center =  M['m10'] / M['m00'], M['m01'] / M['m00']
            #print("Ball mass:", M['m00'])
        else:
            if self.debug:
                print("Ball mass out of mass ranges. (mass={}, lim={})".format(M['m00'], ball_size_lim))
            center = None

        return center

    def processImage(self, frame_number, image):
        try:
            # Downsample the image
            if self.downsample > 1:
                image_dwn = image[::self.downsample, ::self.downsample, :]
                ballmasslim_dwn = self.ballmasslim[0]//(self.downsample**2), self.ballmasslim[1]//(self.downsample**2)
            else:
                image_dwn = image

            center = self.findTheObject(image_dwn, ballmasslim_dwn, mask = self.mask_dwn, denoise = False, name="whole")
            if center is not None:
                center = int(self.downsample*center[0]), int(self.downsample*center[1])

            if center is None:
                if self.debug:
                    print('The ball was not found in the whole image!')
                center_inROI = None
                return None
            else:
                # Find the ball in smaller image
                ROI_xtop = max((center[1]-self.tracking_window_halfsize), 0)
                ROI_xbottom = min((center[1]+self.tracking_window_halfsize), params["resolution"][1])
                ROI_yleft = max((center[0]-self.tracking_window_halfsize), 0)
                ROI_yright = min((center[0]+self.tracking_window_halfsize), params["resolution"][0])
                imageROI = image[ ROI_xtop:ROI_xbottom,  ROI_yleft:ROI_yright, :]

                if self.mask is not None:
                    mask_ROI = self.mask[ROI_xtop:ROI_xbottom,  ROI_yleft:ROI_yright]
                else:
                    mask_ROI = None

                # Find the ball in the region of interest
                center_inROI = self.findTheObject(imageROI, self.ballmasslim, denoise=False, mask=mask_ROI, name="roi")

                # If the ball is not found, raise an exception
                if center_inROI is None:
                    if self.debug:
                        print('The ball was not found in the ROI!')
                    return None
                else:
                    # transform the measured position from ROI to full image coordinates
                    center = ROI_yleft + center_inROI[0], ROI_xtop + center_inROI[1], NAN
            return center
        except Exception as e:
            logger.exception(e)

class TriangleDetector(Detector):
    def __init__(self, **kwargs):
        self.trianglelim = (kwargs["ball_size"][0]/2)**2 * math.pi * 255, (kwargs["ball_size"][1]/2)**2 * math.pi * 255
        self.color_coefs = kwargs["color_coefs"]
        self.downsample = kwargs["downsample"]
        self.threshold = kwargs["threshold"]
        self.tracking_window = kwargs["tracking_window"]
        self.tracking_window_halfsize = self.tracking_window//2
        self.mask = kwargs.get("mask", None)
        if self.mask:
            self.mask_dwn = self.mask[::self.downsample, ::self.downsample]
        else:
            self.mask_dwn = None
        self.name = kwargs["name"]
        self.debug = kwargs.get("debug", 0)
        self.images = {}
        self.orientation_offset = kwargs.get("orientation_offset", 0)

        print("Ball mass must be between {:.0f} px^2 and {:.0f} px^2".format(self.trianglelim[0]/255, self.trianglelim[1]/255))
        print("Image channel combination coefficients: ({})".format(self.color_coefs))

    def __repr__(self):
        return "<{}(color_coefs={}, threshold={})>".format(self.__class__.__name__, self.color_coefs, self.threshold)

    def getImage(self, name):
        return self.images.get(name, None)

    #@profile
    def findTheObject(self, image, triangle_size_lim = None, mask = None, kernel = None, iterations = 2, name = None, compute_rotation = False):
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
        if M['m00'] > 0 and (triangle_size_lim is None  or triangle_size_lim[0] < M['m00'] < triangle_size_lim[1]):
            # Compute the cetner of the object
            center =  M['m10'] / M['m00'], M['m01'] / M['m00']

            # Compute rotation if the corresponding flag is active
            if compute_rotation:
                orientation = self.findOrientation(im_thrs, M, center)

        else:
            if self.debug:
                print("Ball mass out of mass ranges. (mass={}, lim={})".format(M['m00'], triangle_size_lim))

        return center, orientation

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
            # Downsample the image
            if self.downsample > 1:
                image_dwn = image[::self.downsample, ::self.downsample, :]
                trianglelim_dwn = self.trianglelim[0]//(self.downsample**2), self.trianglelim[1]//(self.downsample**2)
            else:
                image_dwn = image

            center, _ = self.findTheObject(image_dwn, trianglelim_dwn, mask = self.mask_dwn, name="whole")
            if center is not None:
                center = int(self.downsample*center[0]), int(self.downsample*center[1])

            if center is None:
                if self.debug:
                    print('The ball was not found in the whole image!')
                return None
            else:
                # Find the ball in smaller image
                ROI_xtop = max((center[1]-self.tracking_window_halfsize), 0)
                ROI_xbottom = min((center[1]+self.tracking_window_halfsize), params["resolution"][1])
                ROI_yleft = max((center[0]-self.tracking_window_halfsize), 0)
                ROI_yright = min((center[0]+self.tracking_window_halfsize), params["resolution"][0])
                imageROI = image[ ROI_xtop:ROI_xbottom,  ROI_yleft:ROI_yright, :]

                if self.mask is not None:
                    mask_ROI = self.mask[ROI_xtop:ROI_xbottom,  ROI_yleft:ROI_yright]
                else:
                    mask_ROI = None

                # Find the ball in the region of interest
                center_inROI, orientation_inROI = self.findTheObject(imageROI, self.trianglelim, mask=mask_ROI, name="roi", compute_rotation = True)

                # If the ball is not found, raise an exception
                if center_inROI is None:
                    if self.debug:
                        print('The ball was not found in the ROI!')
                    return None
                else:
                    # transform the measured position from ROI to full image coordinates
                    return ROI_yleft + center_inROI[0], ROI_xtop + center_inROI[1], math.degrees(orientation_inROI)
        except Exception as e:
            logger.exception(e)