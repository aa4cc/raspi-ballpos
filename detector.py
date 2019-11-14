import find_object
import multi_color_detector
import io
import numpy as np
import cv2
import math
import logging
import warnings
from profilehooks import profile
import colorsys
import pickle
import sys
sys.path
sys.path.insert(0, './find_object')
sys.path.insert(0, './multi_color_detector')

logger = logging.getLogger(__name__)
NAN = float('nan')


class Detector(object):
    def stop(self):
        pass

    def processImage(self, frame_number, image):
        raise RuntimeError("Class {} as child of detector.Detector must override Detector.proccessImage()".format(
            self.__class__.__name__))

    def getImage(self, name):
        return None

    def numberOfObjects(self):
        return 1

    @classmethod
    def fullname(cls):
        if cls.__module__ is None or cls.__module__ == str.__class__.__module__:
            return cls.__name__
        return cls.__module__ + '.' + cls.__name__

    @classmethod
    def from_name(cls, name):
        for c in cls.__subclasses_recursive__():
            if c.fullname() == name:
                return c
        raise KeyError('Class "{}" is not child of "{}", or its not imported'.format(
            name, cls.__name__))

    @classmethod
    def __subclasses_recursive__(cls):
        l = []
        for c in cls.__subclasses__():
            l.append(c)
            l.extend(c.__subclasses_recursive__())
        return l


class ObjectDetector(Detector):
    def __init__(self, **kwargs):
        if "object_size" in kwargs:
            self.objectlim = kwargs["object_size"][0] * \
                255, kwargs["object_size"][1] * 255
        else:
            self.objectlim = None
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
        self.images = {"mask": self.mask}
        self.compute_orientation = kwargs.get('compute_orientation', False)
        self.orientation_offset = kwargs.get("orientation_offset", 0)

        kernel_dwn = kwargs.get("kernel_dwn", None)
        kernel_roi = kwargs.get("kernel_roi", None)

        if kernel_dwn is not None:
            self.kernel_dwn = np.matrix((kernel_dwn), np.uint8)
        else:
            self.kernel_dwn = None

        if kernel_roi is not None:
            self.kernel_roi = np.matrix((kernel_roi), np.uint8)
        else:
            self.kernel_roi = None

        if self.objectlim:
            print("Object mass must be between {:.0f} px^2 and {:.0f} px^2".format(
                self.objectlim[0]/255, self.objectlim[1]/255))
        print("Image channel combination coefficients: ({})".format(self.color_coefs))

    def __repr__(self):
        return "<{}(color_coefs={}, threshold={})>".format(self.__class__.__name__, self.color_coefs, self.threshold)

    @property
    def tracking_window_halfsize(self):
        return self.tracking_window//2

    @property
    def color(self):
        return np.array([255 if c > 0 else 0 for c in self.color_coefs])

    def paint_center(self, image):
        if self.location:
            x, y, t = self.location
            x = round(x)
            y = round(y)

            size = 40
            sx = max(x-(size//2), 0)
            ex = min(x+(size//2), image.shape[1])
            sy = max(y-(size//2), 0)
            ey = min(y+(size//2), image.shape[0])

            image[y, sx:ex, :] = self.color
            image[sy:ey, x, :] = self.color

    def getImage(self, name):
        if name == "center":
            im = self.images.get("image", None)
            if im is None:
                return None
            im = im.copy()
            self.paint_center(im)
            return im

        return self.images.get(name, None)

    # @profile
    def findTheObject(self, image, object_size_lim=None, mask=None, name=None, compute_orientation=False, kernel=None):
        # Initialize orientation and center to None
        orientation = None
        center = None

        # take a linear combination of the color channels to get a grayscale image containg only images of a desired color
        im = np.clip(image[:, :, 0]*self.color_coefs[0] + image[:, :, 1] *
                     self.color_coefs[1] + image[:, :, 2]*self.color_coefs[2], 0, 255).astype(np.uint8)
        self.images[name] = im

        # apply a mask if it is given
        if mask is not None:
            im = cv2.bitwise_and(im, self.im, mask=mask)

        self.images[name+"_masked"] = im

        # threshold the image
        im_thrs = cv2.inRange(im, self.threshold, 255)

        if kernel is not None:
            im_thrs = cv2.erode(im_thrs, kernel, iterations=1)

        # Store the image
        self.images[name+"_thrs"] = im_thrs

        # Compute the moments
        M = cv2.moments(im_thrs)
        # Check whether the area of the found object is within specified limits
        if M['m00'] > 0 and (object_size_lim is None or object_size_lim[0] < M['m00'] < object_size_lim[1]):
            # Compute the cetner of the object
            center = M['m10'] / M['m00'], M['m01'] / M['m00']

            # Compute orientation if the corresponding flag is active
            if compute_orientation:
                orientation = self.findOrientation(im_thrs, M, center)

        else:
            if self.debug:
                print("Ball mass out of mass ranges. (mass={}, lim={})".format(
                    M['m00'], object_size_lim))
            return None

        return center[0], center[1], orientation

    def findOrientation(self, im_thrs, M, center):
        # Detailed description of this algorithm can be found at http://raphael.candelier.fr/?blog=Image%20Moments

        # Central moments (intermediary step)
        a = M['m20']/M['m00'] - center[0]**2  # mu20/m00
        b = 2*(M['m11']/M['m00'] - center[0]*center[1])  # 2*(mu11/m00)
        c = M['m02']/M['m00'] - center[1]**2  # mu02/m00

        # Minor and major axis
        #w = math.sqrt(6*(a+c-math.sqrt(b**2+(a-c)**2)))/2
        #l = math.sqrt(6*(a+c+math.sqrt(b**2+(a-c)**2)))/2

        # Orientation (radians)
        theta = 1/2*math.atan2(b, a-c)

        # Object's pixels coordinates
        j, i = np.nonzero(im_thrs)

        # rot = np.array([-math.sin(theta), math.cos(theta)])
        # coords = np.array([i-center[0], j-center[1]])
        tmpx = (i-center[0])*(-math.sin(theta)) + (j-center[1])*math.cos(theta)
        tmpy = (i-center[0])*(math.cos(theta)) + (j-center[1])*math.sin(theta)

        mx = np.sum(tmpx**3)
        my = np.sum(tmpy**3)

        if abs(mx) > abs(my):
            if mx > 0:
                # Fix direction
                theta += math.pi
        else:
            if my > 0:
                # Fix direction
                theta += math.pi

        theta += self.orientation_offset

        return np.mod(theta, 2*math.pi)

    def processImage(self, frame_number, image):
        try:
            start_y = 0
            end_y = params["resolution"][1]
            start_x = 0
            end_x = params["resolution"][0]

            self.images["image"] = image
            # Downsample the image
            if self.downsample > 1:
                image_dwn = image[::self.downsample, ::self.downsample, :]

                if self.objectlim:
                    object_lim_dwn = self.objectlim[0]//(
                        self.downsample**2), self.objectlim[1]//(self.downsample**2)
                else:
                    object_lim_dwn = None

                location_dwn = self.findTheObject(
                    image_dwn, object_size_lim=object_lim_dwn, mask=self.mask_dwn, name="downsample", kernel=self.kernel_dwn)
                if not location_dwn:
                    if self.debug:
                        print('The object was not found in the whole image!')
                    self.location = None
                    return (None,)

                center = int(
                    self.downsample*location_dwn[0]), int(self.downsample*location_dwn[1])

                halfsize = self.tracking_window_halfsize
                # Find the ball in smaller image
                start_y = max((center[1]-halfsize), 0)
                end_y = min((center[1]+halfsize), params["resolution"][1])
                start_x = max((center[0]-halfsize), 0)
                end_x = min((center[0]+halfsize), params["resolution"][0])

                self.images["image_dwn"] = image_dwn

            imageROI = image[start_y:end_y, start_x:end_x, :]

            self.images["image_roi"] = imageROI

            if self.mask is not None:
                mask_ROI = self.mask[start_y:end_y, start_x:end_x]
                self.images["mask_roi"]
            else:
                mask_ROI = None

            # Find the ball in the region of interest
            location_inROI = self.findTheObject(imageROI, object_size_lim=self.objectlim, mask=mask_ROI,
                                                name="roi", compute_orientation=self.compute_orientation, kernel=self.kernel_roi)
            # If the ball is not found, raise an exception
            if not location_inROI:
                if self.debug:
                    print('The object was not found in the ROI!')
                self.location = None
                return (None,)

            # transform the measured position from ROI to full image coordinates
            x = start_x + location_inROI[0]
            y = start_y + location_inROI[1]
            if location_inROI[2]:
                theta = location_inROI[2]
            else:
                theta = NAN

            self.location = x, y, theta

            return (self.location,)
        except Exception as e:
            logger.exception(e)


class BallDetector(ObjectDetector):
    def __init__(self, **kwargs):
        kwargs["object_size"] = (
            kwargs["ball_size"][0]/2)**2 * math.pi, (kwargs["ball_size"][1]/2)**2 * math.pi
        kwargs["compute_orientation"] = False
        kwargs["orientation_offset"] = 0
        ObjectDetector.__init__(self, **kwargs)


class ObjectDetectorInC(ObjectDetector):
    def processImage(self, frame_number, image):
        im_thrs = np.zeros(image.shape[0:2], 'uint8')
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
        return (location,)

    def getImage(self, name):
        im = self.images.get(name, None)
        if im is None:
            im = self.images.get("thrs", None)
        return im


class BallDetectorInC(ObjectDetectorInC):
    def __init__(self, **kwargs):
        kwargs["object_size"] = (
            kwargs["ball_size"][0]/2)**2 * math.pi, (kwargs["ball_size"][1]/2)**2 * math.pi
        kwargs["compute_orientation"] = False
        kwargs["orientation_offset"] = 0
        ObjectDetector.__init__(self, **kwargs)


'''
A detector that can detect more balls at once, with params in HSV and adjustable via a webpage (/ball_colors). There should never be more than one at once! 
'''
class MultiColorDetector(Detector):
    class BallHSV():
        '''
            class used to represent a ball
            allows the user to specify HSV in the commonly used ways (float and any (0;n) range)
            choose htype = n to input values of H in (0;n) range, same for svtypes
            float assumes input in (0;1)  
        '''
        def __init__(self, h_mid=0, h_tolerance=0, sat_min=1, val_min=1, htype="float", svtypes="float"):
            self.set_new_values(h_mid, h_tolerance, sat_min, val_min, htype=htype, svtypes=svtypes)

        # HSV in 0-1
        def is_color_hsv(self, h, s, v):
            return h > self.h_mid-self.h_tolerance and h < self.h_mid+self.h_tolerance and s > self.sat_min and v > val_min

        # RGB in 0-255
        def is_color_rgb(self, r, g, b):
            h, s, v = colorsys.rgb_to_hsv(
                float(r)/256, float(g)/256, float(b)/256)
            return self.is_color_hsv(h, s, v)

        def get_color_rgb(self, max_value=256):
            return [max_value*x for x in colorsys.hsv_to_rgb(self.h_mid, self.sat_min, self.val_min)]

        def get_color_for_colorpicker(self):
            r,g,b=self.get_color_rgb()
            return "rgb({0:0=3d},{1:0=3d},{2:0=3d})".format(int(r), int(g), int(b))

        def get_color_hexa(self):
            r,g,b=[256*x for x in colorsys.hsv_to_rgb(self.h_mid, 0.8, 0.8)]
            return "#{:0=2X}{:0=2X}{:0=2X}".format(int(r),int(g),int(b))

        def get_color_for_webpage_hidden_input(self):
            return "HSV({0}, {1}, {2})".format(self.h_mid, int(self.sat_min*256), int(self.val_min*256))

        # allows the user to specify HSV in the commonly used ways (float and any (0;n) range)
        def set_new_values(self, h_mid, h_tolerance, sat_min, val_min, htype="float", svtypes="float"):
            if htype.isdigit():
                self.h_mid = float(h_mid)/int(htype)
                self.h_tolerance = float(h_tolerance)/int(htype)
            elif htype == "float":
                self.h_mid = float(h_mid)
                self.h_tolerance = float(h_tolerance)
            else:
                raise RuntimeError(
                    "Unknown htype, choose 'float' or any integer (i.e. '360')")

            if svtypes.isdigit():
                self.sat_min = float(sat_min)/int(svtypes)
                self.val_min = float(val_min)/int(svtypes)
            elif svtypes == "float":
                self.sat_min = float(sat_min)
                self.val_min = float(val_min)
            else:
                raise RuntimeError(
                    "Unknown svtypes, choose 'float' or any integer (i.e. '360')")

            #print(self.h_mid, self.h_tolerance, self.sat_min, self.val_min)

        def hsv_specs(self):
            return self.h_mid, self.h_tolerance, self.sat_min, self.val_min
        
        def __repr__(self):
            return "<BallHSV(h={:.0f}° +-{:.0f}°, s>{:.0f}%, v>{:.0f}%)>".format(self.h_mid*360, self.h_tolerance*360, self.sat_min*100, self.val_min*100)

    def stop(self):
        return

    def numberOfObjects(self):
        return self.number_of_objects

    # necessary function - it initializes the RGB-HSV color table in the C program that is used for detection
    def init_table(self):
         multi_color_detector.init_table([ball.hsv_specs() for ball in self.balls])

    def __init__(self, **kwargs):
        if "object_size" in kwargs:
            self.objectlim = kwargs["object_size"][0] * \
                255, kwargs["object_size"][1] * 255
        else:
            self.objectlim = None

        kernel_dwn = kwargs.get("kernel_dwn", None)
        kernel_roi = kwargs.get("kernel_roi", None)

        if kernel_dwn is not None:
            self.kernel_dwn = np.matrix((kernel_dwn), np.uint8)
        else:
            self.kernel_dwn = None

        if kernel_roi is not None:
            self.kernel_roi = np.matrix((kernel_roi), np.uint8)
        else:
            self.kernel_roi = None

        if self.objectlim:
            print("Object mass must be between {:.0f} px^2 and {:.0f} px^2".format(
                self.objectlim[0]/255, self.objectlim[1]/255))

        self.downsample = kwargs.get("downsample",8)
        self.tracking_window = kwargs["tracking_window"]
        self.mask = kwargs.get("mask", None)
        if self.mask:
            self.mask_dwn = self.mask[::self.downsample, ::self.downsample]
        else:
            self.mask_dwn = None
        self.name = kwargs["name"]
        self.debug = kwargs.get("debug", 0)
        self.images = {"mask": self.mask}
        self.compute_orientation = kwargs.get("compute_orientation", False)
        self.orientation_offset = kwargs.get("orientation_offset", -5.47)


        # MultiColor Specifics
        if params['load_old_color_settings']:
            self.load_color_settings()
            print("Loaded old settings!")
        else:
            ball_colors = kwargs.get("ball_colors", None)
            #print(ball_colors)
            if ball_colors is None:
                print("Error, no colors supplied in JSON")
            self.balls = [self.BallHSV(**x,htype="360", svtypes="100") for x in ball_colors]
        self.number_of_objects = len(self.balls)

        # init the C script
        self.init_table()

    def __repr__(self):
        return("MultiColorDetector")

    def save_color_settings(self):
        with open('color_settings.pkl', 'wb') as settings_file:
            pickle.dump(self.balls, settings_file, pickle.HIGHEST_PROTOCOL)

    def load_color_settings(self):
        try:
            with open('color_settings.pkl', 'rb') as settings_file:
                self.balls=pickle.load(settings_file)
        except: 
            print("Settings file not found!")

    def add_ball(self):
        self.balls.append(self.BallHSV())

    @property
    def tracking_window_halfsize(self):
        return self.tracking_window//2

    @property #legacy
    def color(self):
        # return np.array([255 if c > 0 else 0 for c in self.color_coefs])
        return np.array([0, 0, 0])

    def paint_center(self, image):
        pass
        if self.location:
            x, y, t = self.location
            x = round(x)
            y = round(y)

            size = 40
            sx = max(x-(size//2), 0)
            ex = min(x+(size//2), image.shape[1])
            sy = max(y-(size//2), 0)
            ey = min(y+(size//2), image.shape[0])

            image[y, sx:ex, :] = self.color
            image[sy:ey, x, :] = self.color

    def getImage(self, name):
        if name == "center":
            im = self.images.get("image", None)
            if im is None:
                return None
            im = im.copy()
            self.paint_center(im)
            return im

        return self.images.get(name, None)

    # finds centers of all the balls in the image (if possible)
    def processImage(self, frame_number, image):
        try:
            start_y = 0
            end_y = params["resolution"][1]
            start_x = 0
            end_x = params["resolution"][0]

            self.images["image"] = image
            im_thrs = np.zeros(image.shape[0:2], 'uint8')
            #im=Image.open("image.png")
            window_size=140 if self.compute_orientation else 64
            self.centers = multi_color_detector.process_image(
                image=image,
                image_thrs=im_thrs,
                downsample=self.downsample,
                compute_orientation=self.compute_orientation,
                window_size=window_size,
                orientation_offset=self.orientation_offset
            )
            self.images["im_thrs"]=im_thrs
            return self.centers
        except Exception as e:
            logger.exception(e)
