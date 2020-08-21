#!/usr/bin/env python3
# from numba import jit
import lamp
import processor
import math
import numpy as np
import matplotlib.pyplot as plt
import struct
import logging
import subprocess
#import cv2
from detector import Detector, RansacDetector
from threading import Lock, Thread
from flask import Flask, render_template, send_from_directory, make_response, Response, abort, jsonify, request
from flask_bootstrap import Bootstrap
from flask_colorpicker import colorpicker
import colorsys
from PIL import Image
from io import BytesIO
import re
import time
#import find_object_colors
import inspect
import matplotlib
import warnings
matplotlib.use('Agg')


NaN = float('NaN')

app = Flask(__name__)
Bootstrap(app)
colorpicker(app)
app.processor = None
app.config['SECRET_KEY'] = 'secret!'

app.config['TEMPLATES_AUTO_RELOAD'] = True

app.thread = None

plot_lock = Lock()


def getImage(object=None, type=None):
    if app.processor is None:
        print("App not ready yet")
        return None

    if object is None or object == "processor":
        object = app.processor
    else:
        object = app.processor.getDetector(object)
    return object.getImage(type)


def responseImage(image):
    response = make_response(image)
    response.headers['Content-Type'] = 'image/png'
    return response


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also not to cache the rendered page.
    """
    r.headers["Cache-Control"] = "public, max-age=0, no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


@app.route('/')
@app.route('/detector/<detector>')
def index(detector=None):
    if app.processor is None:
        print("App not ready yet, please retry")
        return 'App not ready yet, please <a href="./">retry</a>'
    if detector is None:
        detectors = app.processor.detectors
    else:
        detectors = filter(lambda d: d.name == detector,
                           app.processor.detectors)
    return render_template("index.html", camera=app.camera, processor=app.processor, params=app.params, detectors=detectors)


@app.route('/centers')
def centers():
    print(app.processor.centers)
    # data = [None if math.isnan(x) else [None if math.isnan(
    #    y) else y for y in x] for x in app.processor.centers]

    data = [None if x is None else [None if math.isnan(
        y) else y for y in x] for x in app.processor.centers]
    return jsonify(data)


@app.route('/restart')
def restart():
    app.processor.restart()
    return 'OK <a href="/">Back</a>'


@app.route('/lamp/on')
def lamp_on():
    lamp.on()
    return 'OK <a href="/">Back</a>'


@app.route('/lamp/off')
def lamp_off():
    lamp.off()
    return 'OK <a href="/">Back</a>'


@app.route('/config')
def config():
    return jsonify(dict(app.params))


@app.route('/config', methods=('POST',))
def config_post():
    data = request.get_json()
    app.params.update(data)
    return jsonify(dict(app.params))


@app.route('/config/loadfile', methods=('POST',))
def config_loadfile():
    filename = request.data
    app.params.load(filename)
    app.processor.restart()
    return jsonify(dict(app.params))


@app.route('/detector/<detector>/threshold/<int:threshold>')
def set_threshold(detector, threshold):
    detectors = filter(lambda d: d.name == detector, app.processor.detectors)

    try:
        next(iter(detectors)).threshold = threshold
    except StopIteration:
        abort(404)

    return "Ok"


@app.route('/image')
@app.route('/image/<object>')
@app.route('/image/<object>/<type>')
def image(object=None, type=None):
    image = getImage(object, type)
    if image is None:
        abort(404)
        return "Not loaded yet"
    if len(image.shape) == 3:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # _, buffer = cv2.imencode('.png', image)
        pil_image=Image.fromarray(image)
        byteIO=BytesIO()
        pil_image.save(byteIO,format='PNG')
    return responseImage(byteIO.getvalue())


@app.route('/imagesc')
@app.route('/imagesc/<object>')
@app.route('/imagesc/<object>/<type>')
def imagesc(object=None, type=None):
    image = getImage(object, type)
    if image is None:
        abort(404)
        return "Not loaded yet"
    buffer = BytesIO()
    with plot_lock:
        if len(image.shape) == 3:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='Greys',  interpolation='nearest')
            plt.colorbar()
        plt.savefig(buffer)
        plt.close()
    return responseImage(buffer.getvalue())


@app.route('/wb')
@app.route('/wb/<int:step>')
def wb(step=0):
    return render_template("wb.html", camera=app.camera, processor=app.processor, params=app.params, step=step)


@app.route('/wb/value')
def wb_value():
    image = getImage()
    # print(image[:,:,0].shape)
    if image is None:
        abort(404)
        return "Not loaded yet"
    image=np.array(image)
    pt1 = tuple(int(v/2-50) for v in app.params["resolution"])
    pt2 = tuple(int(v/2+50) for v in app.params["resolution"])
    [b,g,r]=np.dsplit(image,image.shape[-1])
    # print(b.shape)
    #jsonify(cv2.mean(image[pt1[0]:pt2[0], pt1[1]:pt2[1], :]))
    # print((list(map(float(np.mean([image[:][:][i] for i in range(3)], axis=0))))))
    return jsonify([np.mean(b),np.mean(g),np.mean(r)])


@app.route('/wb/value/<int:a>,<int:b>')
@app.route('/wb/value/<float:a>,<int:b>')
@app.route('/wb/value/<int:a>,<float:b>')
@app.route('/wb/value/<float:a>,<float:b>')
def wb_set(a, b):
    print(a,b)
    # image_wb()
    app.camera.awb_gains = (a, b)
    return "OK"


@app.route('/image/wb')
def image_wb():
#     image = getImage("processor", "centers")


    pt1 = tuple(int(v/2-50) for v in app.params["resolution"])
    pt2 = tuple(int(v/2+50) for v in app.params["resolution"])

    im = getImage()
    if im is None:
        abort(404)
    pil_image=Image.fromarray(im)
    byteIO=BytesIO()
    pil_image.save(byteIO,format='PNG')
    return responseImage(byteIO.getvalue())


def get_hsv_detector():
    if app.processor is None:
        return "App processor hasn't loaded yet!"
    if len(app.processor.detectors) < 0:
        return "No detector registered!"
    for used_detector in app.processor.detectors:
        # type didn't work for some reason...
        used_str=str(used_detector)
        if used_str == "MultiColorDetector" or used_str=='RansacDetector':
            return app.processor.getDetector(used_detector.name)
    return "This webpage only works if you're using the MultiDetector..."

# main ball_colors UI webpage, computes some statistics, prepares files for other functions and creates the website itself
@app.route('/ball_colors')
def ball_colors():
    # check if everything has been loaded
    im = getImage()

    if im is None:
        return "Program hasn't properly started yet - try it again in a few seconds. :-)"

    MultiDetector = get_hsv_detector()
    if not isinstance(MultiDetector, Detector):
        return MultiDetector

    # get samples
    centers = []
    # print("1")

    frame_number = app.processor.frame_number
    test_iterations = request.args.get("i")
    if test_iterations is None:
        test_iterations = 5
    else:
        try:
            test_iterations = int(test_iterations)
            if test_iterations < 1:
                test_iterations = 1
        except:
            print("Test iteration must be integer! Setting default value...")
            test_iterations = 5
    for i in range(test_iterations):
        centers.append(app.processor.centers)
        while app.processor.frame_number == frame_number: #wait for next frame
            continue
        frame_number = app.processor.frame_number

    balls = [[x[i] for x in centers]
             for i in range(len(centers[0]))]  # change data structure
    x_coordinates = [[]for i in range(len(balls))]
    y_coordinates = [[]for i in range(len(balls))]
    thetas = [[]for i in range(len(balls))]
    ball_centers_found = []

    # compute statistics
    for ball_index, ball in enumerate(balls):
        current_centers_found = 0
        for sample in ball:
            if sample is not None and sample[0] is not None:
                current_centers_found += 1
                x_coordinates[ball_index].append(sample[0])
                y_coordinates[ball_index].append(sample[1])
            if sample is not None and sample[2] is not None:
                thetas[ball_index].append(sample[2])
        # if current_centers_found == 0:
        #     x_coordinates.append(np.nan)
        #     y_coordinates.append(np.nan)
        ball_centers_found.append(current_centers_found)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ball_center_means = [(np.mean(x_coordinates[i]), np.mean(
            y_coordinates[i]), np.mean(thetas[i])) for i in range(len(balls))]
        ball_center_stds = [(np.std(x_coordinates[i]), np.std(
            y_coordinates[i]), np.std(thetas[i])) for i in range(len(balls))]
        percentages = [100*ball_centers_found[i] /
                    test_iterations for i in range(len(balls))]

    
    # save images for website
    pil_image=Image.fromarray(im)
    # pil_image.save('static/imageBRG.png')
    # pil_image=pil_image.convert('BGR;16')
    pil_image.save('static/image.png')
    # cv2.imwrite('static/imageBRG.png', im)
    
    # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('static/image.png', im)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # pil_image.save('static/imageBRG.png')
    pil_image=pil_image.convert('HSV')
    np.save('static/image_hsv', np.array(pil_image))

    return render_template('ball_colors.html', balls=MultiDetector.balls, found=ball_centers_found, test_iterations=test_iterations, means=ball_center_means, percentages=percentages, stds=ball_center_stds, int=int, len=len)

# receives x,y coordinates and responds with picture color in RGB at that position
@app.route('/ball_colors/color')
def color():
    try:
        x = int(request.args.get('x'))
        y = int(request.args.get('y'))
        #print(x, y)
    except:
        print("Error: X,Y not int!")
        x = 0
        y = 0
    if(x < 0 or x > 480):
        print("Error: x out of bounds!")
        x = 0
    if(y < 0 or y > 480):
        print("Error: y out of bounds")
        y = 0
    im = np.load('static/image_hsv.npy')
    pixel = im[y, x]
    # print(pixel*[360/256,100/256,100/256])
    r, g, b = colorsys.hsv_to_rgb(float(pixel[0])/256, 0.3, 0.3)

    #print("R: {}, G: {}, B: {}".format(int(r*256), int(g*256), int(b*256)))
    return jsonify(r=int(r*256), g=int(g*256), b=int(b*256))


'''
Used to change ball colors. However, this does not apply the changes - merely set them. 
It is still necessary to reinit the table in C, preferably using '/ball_colors/set_colors'
'''

# @jit(nopython=True, cache=True)
from ransac_detector_ctypes import Ball_t
from ctypes import byref
def compute_mask(im, detector, ball_t):
    image=getImage()
    mask=np.zeros(shape=tuple(image.shape[:2]),dtype=np.uint8)
    # print(mask.shape)
    if isinstance(detector,RansacDetector):
        detector.c_funcs.get_segmentation_mask(im,*image.shape[:2],mask,byref(ball_t))
    else:
        print("This detector does not support segmentation mask in browser, sorry!")
    return mask

@app.route('/ball_colors/limits')
def limits():
    h_mod=256 #180
    try:
        formatted = (request.args.get('formatted'))
        tolerance = int(request.args.get('tolerance'))
        index = int(request.args.get('index'))
        m = re.match(r'HSV\((.*),(.*),(.*)\)', formatted)
        h = float(m.group(1))*h_mod #supplied as float in [0,1]
        #tolerance /= 2  # from 360 to 180
        s = int(m.group(2)) #supplied as int in [0,255]
        v = int(m.group(3)) #supplied as int in [0,255]
    except:
        print("Error formatting at /limits, received: formatted: {}, tolerance: {}, index: {}".format(
              formatted, request.args.get('tolerance'), request.args.get('index')))
        return "ERROR"
    # TODO: check if hues overlap
    h_min = int(h-tolerance)
    h_max = int(h+tolerance)
    sat_min = int(s)
    v_min = int(v)

    if h_min < 0:
        h_min += h_mod
        h_max += h_mod

    im = np.load('static/image_hsv.npy')
    lower_bound = np.array([h_min, sat_min, v_min])
    upper_bound = np.array([h_max, 255, 255])

    MultiDetector = get_hsv_detector()
    if not isinstance(MultiDetector, Detector):
        return MultiDetector

    print(f"Webpage: {h}, {tolerance} {s}, {v}")
    MultiDetector.balls[index].set_new_values(
        h, tolerance, s, v, htype="256", svtypes="256")

    # start=time.time()

    mask=compute_mask(im, MultiDetector, Ball_t(h_min,h_max,s,v))

    # print(compute_mask.inspect_types())
    # print(f"Took {time.time()-start}")
    image = Image.fromarray(mask)
    image.save("static/im_thrs-{}.png".format(index))
    return "OK"


@app.route('/colorpicker-master/<path:path>')
def colorpicker_plugin(path):
    return send_from_directory('colorpicker-master', path)

# used to save new settings
@app.route('/ball_colors/set_colors')
def set_colors():
    MultiDetector = get_hsv_detector()
    if not isinstance(MultiDetector, Detector):
        return MultiDetector
    MultiDetector.init_table()
    MultiDetector.save_color_settings()
    return "OK"

@app.route('/ransac')
def ransac_settings():
    detector=get_hsv_detector()
    if not isinstance(detector,RansacDetector):
        return "This page only works with RansacDetector"
    centers=detector.centers
    center=centers[0]
    print(center)
    if center is None:
        return "Ball not found"
    image=getImage()
    offset=30
    w_low=max(0,int(center[0]-offset))
    w_high=min(image.shape[0],int(center[0]+offset))

    h_low=max(0,int(center[1]-offset))
    h_high=min(image.shape[1],int(center[1]+offset))
    
    image_crop=image[h_low:h_high,w_low:w_high,:]
    images=[image_crop]
    images.extend(detector.processImageOverlay(image_crop,[offset,offset]))
    images_strs=["image_crop","seg_background","seg_ball","seg_border","ransac_contour","ransac_tolerance_contour","lsq_modeled_contour","lsq_border_contour"]
    checkbox_labels=["","Background mask", "Ball mask", "Border mask", "Ransac fit", "Ransac tolerance (\"modeled\" pixels)","LSQ (fit to RANSAC)","LSQ (fit to all border)"]

    for image, image_str in zip(images,images_strs):
        image=Image.fromarray(image)
        image.save(f"static/{image_str}-{0}.png")

    return render_template('ransac.html',ball_nr=detector.number_of_objects,images_n_labels=list(zip(images_strs,checkbox_labels)))

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy

def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def weighted_line(r0, c0, r1, c1, w=2, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    r0=int(r0)
    c0=int(c0)
    r1=int(r1)
    c1=int(c1)
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

def paint_triangle(im, top_left_coords, bottom_left_coords, bottom_right_coords):
    if top_left_coords is not None and bottom_right_coords is not None:
        theta=np.arctan2(top_left_coords[1]-bottom_right_coords[1],top_left_coords[0]-bottom_right_coords[0])+np.pi/4
        center_of_mass=[(top_left_coords[0]+bottom_right_coords[0]+bottom_left_coords[0])/3,(top_left_coords[1]+bottom_right_coords[1]+bottom_left_coords[1])/3]
        print(center_of_mass)
        for i in range(-4,4+1):
            im[int(center_of_mass[1])+i,int(center_of_mass[0]),1]=0xff
            im[int(center_of_mass[1]),int(center_of_mass[0])+i,1]=0xff
        
        # paint the triangle
        side_length_pixel=122

        bl_corner=[center_of_mass[0]-side_length_pixel//3,center_of_mass[1]-side_length_pixel//3] # BL - bottom-left
        br_corner=[bl_corner[0]+side_length_pixel,bl_corner[1]] # BR - bottom-right
        tl_corner=[bl_corner[0],bl_corner[1]+side_length_pixel] # TL - top-left

        bl_corner=rotate(center_of_mass,bl_corner,theta)
        br_corner=rotate(center_of_mass,br_corner,theta)
        tl_corner=rotate(center_of_mass,tl_corner,theta)

        lines=[weighted_line(bl_corner[0],bl_corner[1],br_corner[0],br_corner[1]),
            weighted_line(tl_corner[0],tl_corner[1],br_corner[0],br_corner[1]),
            weighted_line(bl_corner[0],bl_corner[1],tl_corner[0],tl_corner[1])]
        for line in lines:
            for i in range(len(line[0])):
                im[line[1][i],line[0][i]]=[0xff,0x33,0xda]#int(150*line[2][i])
        return im


@app.route('/triangle')
def triangle():
    # check if everything has been loaded
    im = getImage()
    if im is None:
        return "Program hasn't properly started yet - try it again in a few seconds. :-)"
    print(app.processor.centers)
    centers=app.processor.centers
    red=centers[0]
    blue=centers[1]
    green=centers[2]
    pink=centers[3]
    yellow=centers[4]
    orange=centers[5]
    
    # paint centers
    for center in centers:
        if center is not None:
            for i in range(-16,16):
                if center[0]+i<0 or center[0]+i>im.shape[0] or center[1]+i<0 or center[1]+i>im.shape[1]:
                    continue
                im[int(center[1])+i,int(center[0]),:]=0xff
                im[int(center[1]),int(center[0])+i,:]=0xff

    # paint triangles
    paint_triangle(im,blue,yellow, red)
    paint_triangle(im,pink, orange, green)

    # return the data as a png
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', image)
    return responseImage(buffer.tobytes())

def start():
    if not app.thread:
        app.thread = Thread(target=app.run, daemon=True, kwargs={
                            "host": "0.0.0.0", "port": 5001, "debug": True, "use_reloader": False})
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        app.thread.start()


app.start = start

if __name__ == '__main__':
    start()
