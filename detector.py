import io
import numpy as np
import cv2
import math
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Detector(object):
    pass


class Simple(Detector):
    def __init__(self, **kwargs):
        self.ballmasslim = (kwargs["ball_size"][0]/2)**2 * math.pi * 255, (kwargs["ball_size"][1]/2)**2 * math.pi * 255
        self.color_coefs = kwargs["color_coefs"]
        self.downsample = kwargs["downsample"]
        self.threshold = kwargs["threshold"]
        self.tracking_window = kwargs["tracking_window"]
        self.tracking_window_halfsize = self.tracking_window//2
        self.processor = kwargs["processor"]
        self.name = kwargs["name"]
        self.debug = kwargs.get("debug", 0)

        print("Ball mass must be between {:.0f} px^2 and {:.0f} px^2".format(self.ballmasslim[0]/255, self.ballmasslim[1]/255))
        print("Image channel combination coefficients: ({})".format(self.color_coefs))

    def findTheBall(self, image, ball_size_lim = None, mask = None, denoise = True, kernel = None, iterations = 2):
        # take a linear combination of the color channels to get a grayscale image containg only images of a desired color
        im = np.clip(image[:,:,0]*self.color_coefs[0] + image[:,:,1]*self.color_coefs[1] + image[:,:,2]*self.color_coefs[2], 0, 255).astype(np.uint8) 

        if self.debug ==1:
            plt.imshow(im)
            plt.colorbar()
            plt.show()

        # apply a mask if it is given
        if mask is not None:
            im = cv2.bitwise_and(im, im, mask = mask)

        im_thrs = cv2.inRange(im, self.threshold,255)
        if denoise:
            # im_denoised = cv2.dilate(im_thrs, kernel, iterations)
            # im_denoised = cv2.erode(im_denoised, kernel, iterations)
            im_denoised = cv2.morphologyEx(im_thrs, cv2.MORPH_OPEN, None)
        else:
            im_denoised = im_thrs

        # cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # ((x, y), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(im_denoised)
        if M['m00'] > 0 and (ball_size_lim is None  or ball_size_lim[0] < M['m00'] < ball_size_lim[1]):
            center = ( (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) )
        else:
            print("Ball mass out of mass ranges.")
            center = None

        return center, im_thrs, im_denoised

    def processImage(self, frame_number, image):
        try:
            # Downsample the image
            if self.downsample > 1:
                image_dwn = image[::self.downsample, ::self.downsample, :]
                ballmasslim_dwn = self.ballmasslim[0]//(self.downsample**2), self.ballmasslim[1]//(self.downsample**2)
            else:
                image_dwn = image

            center, im_thrs, im_denoised = self.findTheBall(image_dwn, ballmasslim_dwn, mask = self.processor.mask_dwn, denoise = False)
            if center is not None:
                center = (self.downsample*center[0], self.downsample*center[1])


            # Save the the region of interest image if the debug option is chosen, or ball not found
            if params['debug'] > 1 or (center is None and params["debug"]):
                # if center is not None:
                    # cv2.circle(img_dwnsample, center, 5, (0, 0, 255), -1)
                cv2.imwrite("{}im_dwn_denoised{}.png".format(params['img_path'], frame_number), im_denoised)
                cv2.imwrite("{}image{}.png".format(params['img_path'], frame_number), image)       
                cv2.imwrite("{}im_thrs{}.png".format(params['img_path'], frame_number), im_thrs)         

            if center is None:
                print('The ball was not found in the whole image!')
                center_inROI = None
            else:
                # Find the ball in smaller image
                ROI_xtop = max((center[1]-self.tracking_window_halfsize), 0)
                ROI_xbottom = min((center[1]+self.tracking_window_halfsize), params["resolution"][1])
                ROI_yleft = max((center[0]-self.tracking_window_halfsize), 0)
                ROI_yright = min((center[0]+self.tracking_window_halfsize), params["resolution"][0])
                imageROI = image[ ROI_xtop:ROI_xbottom,  ROI_yleft:ROI_yright, :]

                if self.processor.mask is not None:
                    mask_ROI = self.proccesor.mask[ROI_xtop:ROI_xbottom,  ROI_yleft:ROI_yright]
                else:
                    mask_ROI = None

                # Find the ball in the region of interest
                center_inROI, im_thrs, im_denoised = self.findTheBall(imageROI, self.ballmasslim, denoise=False, mask=mask_ROI)

                # If the ball is not found, raise an exception
                if center_inROI is None:
                    print('The ball was not found in the ROI!')
                else:
                    # transform the measured position from ROI to full image coordinates
                    center = (ROI_yleft + center_inROI[0], ROI_xtop + center_inROI[1])

            # Save the the region of interest image if the debug option is chosen
            if params['debug']:
                # cv2.imwrite("{}im_thrs%d.png".format(params['img_path'], frame_number), im_thrs)

                if center_inROI is not None:
                    cv2.circle(imageROI, center_inROI, 5, (0, 0, 255), -1)
                    cv2.imwrite("{}im_roi{}.png".format(params['img_path'], frame_number), imageROI) 
                    cv2.imwrite("{}im_denoised{}.png".format(params['img_path'], frame_number), im_denoised)         
            return center
        except Exception as e:
            logger.exception(e)