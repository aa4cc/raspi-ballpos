#!/usr/bin/env python3

import lamp
import processor
import math
import numpy as np
import matplotlib.pyplot as plt
import struct
import logging
import subprocess
import cv2
from detector import Detector
from threading import Lock, Thread
from flask import Flask, render_template, send_from_directory, make_response, Response, abort, jsonify, request
from flask_bootstrap import Bootstrap
from flask_colorpicker import colorpicker
import colorsys
from PIL import Image
from io import BytesIO
import re
import time
import find_object_colors
import inspect
import matplotlib
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
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', image)
    return responseImage(buffer.tobytes())


@app.route('/imagesc')
@app.route('/imagesc/<object>')
@app.route('/imagesc/<object>/<type>')
def imagesc(object=None, type=None):
    image = getImage(object, type)
    if image is None:
        abort(404)
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
    if image is None:
        abort(404)
    pt1 = tuple(int(v/2-50) for v in app.params["resolution"])
    pt2 = tuple(int(v/2+50) for v in app.params["resolution"])
    return jsonify(cv2.mean(image[pt1[0]:pt2[0], pt1[1]:pt2[1], :]))


@app.route('/wb/value/<int:a>,<int:b>')
@app.route('/wb/value/<float:a>,<int:b>')
@app.route('/wb/value/<int:a>,<float:b>')
@app.route('/wb/value/<float:a>,<float:b>')
def wb_set(a, b):
    print(a,b)
    app.camera.awb_gains = (a, b)
    return "OK"


@app.route('/image/wb')
def image_wb():
    image = getImage("processor", "centers")
    if image is None:
        abort(404)

    pt1 = tuple(int(v/2-50) for v in app.params["resolution"])
    pt2 = tuple(int(v/2+50) for v in app.params["resolution"])
    image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), 5)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return responseImage(buffer.tobytes())


def get_multidetector():
    if app.processor is None:
        return "App processor hasn't loaded yet!"
    if len(app.processor.detectors) < 0:
        return "No detector registered!"
    for used_detector in app.processor.detectors:
        # type didn't work for some reason...
        if str(used_detector) == "MultiColorDetector":
            return app.processor.getDetector(used_detector.name)
    return "This webpage only works if you're using the MultiDetector..."

# main ball_colors UI webpage, computes some statistics, prepares files for other functions and creates the website itself
@app.route('/ball_colors')
def ball_colors():
    # check if everything has been loaded
    im = getImage()
    if im is None:
        return "Program hasn't properly started yet - try it again in a few seconds. :-)"

    MultiDetector = get_multidetector()
    if not isinstance(MultiDetector, Detector):
        return MultiDetector

    # get samples
    centers = []
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
        while app.processor.frame_number == frame_number:
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
        '''if current_centers_found == 0:
            x_coordinates.append(np.nan)
            y_coordinates.append(np.nan)'''
        ball_centers_found.append(current_centers_found)
    ball_center_means = [(np.mean(x_coordinates[i]), np.mean(
        y_coordinates[i]), np.mean(thetas[i])) for i in range(len(balls))]
    ball_center_stds = [(np.std(x_coordinates[i]), np.std(
        y_coordinates[i]), np.std(thetas[i])) for i in range(len(balls))]
    percentages = [100*ball_centers_found[i] /
                   test_iterations for i in range(len(balls))]

    # save images for website
    cv2.imwrite('static/imageBRG.png', im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite('static/image.png', im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    np.save('static/image_hsv', im)

    return render_template('ball_colors.html', balls=MultiDetector.balls, found=ball_centers_found, test_iterations=test_iterations, means=ball_center_means, percentages=percentages, stds=ball_center_stds, int=int, len=len)

# receives x,y coordinates and reponds with picture color in RGB at that position
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
    r, g, b = colorsys.hsv_to_rgb(float(pixel[0])/180, 0.3, 0.3)

    #print("R: {}, G: {}, B: {}".format(int(r*256), int(g*256), int(b*256)))
    return jsonify(r=int(r*256), g=int(g*256), b=int(b*256))


'''
Used to change ball colors. However, this does not apply the changes - merely set them. 
It is still necessary to reinit the table in C, preferably using '/ball_colors/set_colors'
'''
@app.route('/ball_colors/limits')
def limits():
    try:
        formatted = (request.args.get('formatted'))
        tolerance = int(request.args.get('tolerance'))
        index = int(request.args.get('index'))
        m = re.match(r'HSV\((.*),(.*),(.*)\)', formatted)
        h = float(m.group(1))*180
        tolerance /= 2  # from 360 to 180
        s = int(m.group(2))
        v = int(m.group(3))
    except:
        print("Error formatting at /limits, received: formatted: {}, tolerance: {}, index: {}".format(
              formatted, request.args.get('tolerance'), request.args.get('index')))
        return "ERROR"

    h_min = int(h-tolerance)
    h_max = int(h+tolerance)
    sat_min = int(s)
    v_min = int(v)

    if h_min < 0:
        h_min += 180
        h_max += 180

    im = np.load('static/image_hsv.npy')
    lower_bound = np.array([h_min, sat_min, v_min])
    upper_bound = np.array([h_max, 255, 255])
    if h_max > 180:
        mask1 = cv2.inRange(im, lower_bound, np.array(
            [180, 255, 255]))  # (h_min,180)
        mask2 = cv2.inRange(im, np.array([0, sat_min, v_min]), np.array(
            [h_max % 180, 255, 255]))  # (0,h_max%180)
        im = mask1 | mask2
    else:
        im = cv2.inRange(im, lower_bound, upper_bound)

    MultiDetector = get_multidetector()
    if not isinstance(MultiDetector, Detector):
        return MultiDetector

    MultiDetector.balls[index].set_new_values(
        h, tolerance, s, v, htype="180", svtypes="256")

    image = Image.fromarray(im)
    image.save("static/im_thrs-{}.png".format(index))
    return "OK"


@app.route('/colorpicker-master/<path:path>')
def colorpicker_plugin(path):
    return send_from_directory('colorpicker-master', path)

# used to save new settings
@app.route('/ball_colors/set_colors')
def set_colors():
    MultiDetector = get_multidetector()
    if not isinstance(MultiDetector, Detector):
        return MultiDetector
    MultiDetector.init_table()
    MultiDetector.save_color_settings()
    return "OK"

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

@app.route('/triangle')
def draw_triangle():
    # check if everything has been loaded
    im = getImage()
    print((im).shape)
    if im is None:
        return "Program hasn't properly started yet - try it again in a few seconds. :-)"
    print(app.processor.centers)
    centers=app.processor.centers
    yellow=centers[4]
    blue=centers[1]
    red=centers[5]


    # paint centers
    for center in centers:
        if center is not None:
            for i in range(-16,16):
                if center[0]+i<0 or center[0]+i>im.shape[0] or center[1]+i<0 or center[1]+i>im.shape[1]:
                    continue
                im[int(center[1])+i,int(center[0]),:]=0xff
                im[int(center[1]),int(center[0])+i,:]=0xff

    if blue is not None and red is not None:
        theta=np.arctan2(blue[1]-red[1],blue[0]-red[0])+np.pi/4
        center_of_mass=[(blue[0]+red[0]+yellow[0])/3,(blue[1]+red[1]+yellow[1])/3]
        print(center_of_mass)
        for i in range(-4,4+1):
            im[int(center_of_mass[1])+i,int(center_of_mass[0]),1]=0xff
            im[int(center_of_mass[1]),int(center_of_mass[0])+i,1]=0xff
        
        # paint the triangle
        side_length_pixel=122

        y_corner=[center_of_mass[0]-side_length_pixel//3,center_of_mass[1]-side_length_pixel//3]
        r_corner=[y_corner[0]+side_length_pixel,y_corner[1]]
        b_corner=[y_corner[0],y_corner[1]+side_length_pixel]

        y_corner=rotate(center_of_mass,y_corner,theta)
        r_corner=rotate(center_of_mass,r_corner,theta)
        b_corner=rotate(center_of_mass,b_corner,theta)

        lines=[weighted_line(y_corner[0],y_corner[1],r_corner[0],r_corner[1]),
            weighted_line(b_corner[0],b_corner[1],r_corner[0],r_corner[1]),
            weighted_line(y_corner[0],y_corner[1],b_corner[0],b_corner[1])]
        for line in lines:
            for i in range(len(line[0])):
                im[line[1][i],line[0][i],:]=int(255*line[2][i])

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
