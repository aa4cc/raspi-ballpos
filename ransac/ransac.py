import sys
import random
from PIL import Image, ImageDraw
import numpy as np
import math
import copy
import time
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True, cache=True)
def rgb2hsv(r, g, b):
    # converts RGB color to HSV
    # normalize the values to [0,1]
    r_norm = r/255
    g_norm = g/255
    b_norm = b/255
    # print(r_norm,g_norm,b_norm)

    # perform the conversion based on an algorithm found online
    c_max = max([r_norm, g_norm, b_norm])
    c_min = min([r_norm, g_norm, b_norm])
    delta = c_max-c_min

    if delta <= 1e-4:
        h = 0
    elif c_max == r_norm:
        h = ((g_norm-b_norm)/delta)
    elif c_max == g_norm:
        h = ((b_norm-r_norm)/delta+2)
    elif c_max == b_norm:
        h = ((r_norm-g_norm)/delta+4)
    h *= 60

    s = 0 if c_max == 0 else delta/c_max*100
    v = c_max*100

    return h, s, v


@jit(nopython=True, cache=True)
def hsv_in_range(h, s, v, h_min, h_max, s_min, v_min, h_mod):
    # h_min < h_max (h_max may be >h_mod)
    if h_max > h_mod:
        return (h < h_max % h_mod or h > h_min) and s > s_min and v > v_min
    else:
        return h < h_max and h > h_min and s > s_min and v > v_min


@jit(nopython=True, cache=True)
def get_ball_colors(table, h_min, h_max, s_min, v_min, h_mod=360, color_res=256):
    # returns a True/False array whether an RGB pixel is within a HSV rectangle
    ball_colors = np.empty(
        shape=(color_res, color_res, color_res), dtype=np.dtype('?'))
    for r in range(color_res):
        for g in range(color_res):
            for b in range(color_res):
                h, s, v = table[r, g, b]
                ball_colors[r, g, b] = hsv_in_range(
                    h, s, v, h_min, h_max, s_min, v_min, h_mod)
    return ball_colors


@jit(nopython=True, cache=True)
def get_rgb2hsv_table(color_res=256):
    # computes a conversion table from RGB to HSV
    table = np.empty(shape=(256, 256, 256, 3), dtype=np.uint16)
    for r in range(color_res):
        for g in range(color_res):
            for b in range(color_res):
                table[r, g, b, :] = rgb2hsv(r, g, b)
    return table


@jit(nopython=True, cache=True)
def get_intersections(x0, y0, r0, x1, y1, r1):
    # finds intersections of the two circles specified
    # https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt((x1-x0)**2 + (y1-y0)**2)

    # non intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0**2-r1**2+d**2)/(2*d)
        h = math.sqrt(r0**2-a**2)
        x2 = x0+a*(x1-x0)/d
        y2 = y0+a*(y1-y0)/d
        x3 = x2+h*(y1-y0)/d
        y3 = y2-h*(x1-x0)/d

        x4 = x2-h*(y1-y0)/d
        y4 = y2+h*(x1-x0)/d

        return [(x3, y3), (x4, y4)]


@jit(nopython=True, cache=True)
def get_segmentation_mask(img, ball_colors, step=2):
    # finds which pixels are balls and which are background (output as a width x height True/False array, possibly could be updated to uint8 to find more colors)
    width, height = img.shape[:2]
    mask = np.zeros((width, height), dtype=np.uint8)
    mask_coords = []
    for x in range(0, width, step):
        for y in range(0, height, step):
            # pixel=tuple(img[x,y,:])  does not work due to numba unfortunately
            r = img[x, y, 0]
            g = img[x, y, 1]
            b = img[x, y, 2]
            pixel = (r, g, b)
            if ball_colors[r, g, b]:
                mask[y, x] = 1
                mask_coords.append((y, x))
    if len(mask_coords)==0:
        # naughty hack to that numba doesnt crash (it does not like when a function returns an empty list)
        mask_coords.append((-1,-1))
    return mask, np.array(mask_coords)


@jit(nopython=True, cache=True)
def get_border_mask(segmentation_mask, ball_coords, step=2):
    # returns border pixels, i.e. pixels with both background and balls next to them
    # this function is about 10x faster than ransac and get_segmentation, so no need to over-engineer it
    width, height = segmentation_mask.shape
    border_mask = np.zeros((width, height), dtype=np.uint8)
    border_coords = []  # List()
    for i in range(len(ball_coords)):
        x, y = ball_coords[i]
        neighbours = []
        # look in all 8 directions
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x_index = x+dx*step
                y_index = y+dy*step
                if x_index <= 0 or x_index >= width or y_index < 0 or y_index >= height:
                    # print("Out of bounds")
                    continue
                else:
                    neighbours.append(segmentation_mask[x_index][y_index])

        # 1 ~ ball, 0 ~ background
        if 1 in neighbours and 0 in neighbours:
            border_mask[x, y] = 1
            border_coords.append((x, y))
    if len(border_coords)==0:
        border_coords.append((-1,-1)) # again, naughty hack so that numba does not crash
    return border_mask, np.array(border_coords)


@jit(nopython=True, cache=True)#,fastmath=True)
def norm_axis1(differences):
    # numba does not support the axis argument for np.linalg.norm, so a custom function is unfortunately necessary
    power = np.power(differences, 2).T
    ret = np.sqrt(power[0]+power[1])
    return ret


@jit(nopython=True, cache=True)
def filter_detected_ball(center, r, border_coords):
    # removes pixels related to the ball located at center
    differences = border_coords-center
    distances = norm_axis1(differences)
    new_bcoords = []
    for i, dist in enumerate(distances):
        if dist > 1.2*r:
            new_bcoords.append((border_coords[i, 0], border_coords[i, 1]))
        if len(new_bcoords):
            new_bcoords.append((-1,-1)) # once more, let's bribe numba to play nice and not crash
    return np.array(new_bcoords)

@jit(nopython=True, cache=True)
def ransac(img, ball_colors, confidence_thrs=50, max_iter=40, nr_of_objects=2, ball_diameter=40, verbous=False):
    # performs the ransac algorithm
    # finds ball(s) in img, colors are specified in ball_colors, which is a table 256x256x256 (RGB) denoting which colors are to be considered for ball
    # runs until either confence_thrs pixels are fitting the model (border) or when max_iter is reached
    # ball diameter (in pixels) defines the model that is fitted

    r = ball_diameter/2

    step = 2 # only every 2nd pixel is checked (for speed)
    segmentation_mask, ball_coords = get_segmentation_mask(
        img, ball_colors, step)
    border_mask, border_coords = get_border_mask(
        segmentation_mask, ball_coords, step)
    ball_centers = []

    # try to find as many objects as specified
    for _ in range(nr_of_objects):
        # check if there even are enough border pixels to look through
        if len(border_coords) < confidence_thrs/4:
            if verbous:
                print("Out of balls")
            ball_centers.append(None)
            continue
        best_model = np.array([0, 0])
        best_inliers = 0
        for i in range(max_iter):
            # pick random two border points and check the two circles that go through
            intersections = None
            while intersections is None:
                index1 = np.random.randint(len(border_coords)-1)
                index2 = index1
                while index2 == index1:
                    index2 = np.random.randint(len(border_coords)-1)
                point1 = border_coords[index1]
                point2 = border_coords[index2]
                intersections = get_intersections(
                    point1[0], point1[1], r, point2[0], point2[1], r)
            # list(map(...)) does not work in numba...
            center1 = (int(intersections[0][0]), int(intersections[0][1]))
            center2 = (int(intersections[1][0]), int(intersections[1][1]))
            if center1 == center2:
                center_list = [center1]
            else:
                center_list = [center1, center2]
            
            # count border pixels fitting the model
            for center in center_list:
                npcenter = np.array(center)
                differences = (border_coords-npcenter)
                # numba does not support the axis argument --> custom function
                distances = norm_axis1(differences)
                inliers = len(
                    distances[np.where((distances < 1.05*r) & (distances > 0.95*r))])
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_model = npcenter
            if best_inliers >= confidence_thrs:
                if verbous:
                    print("Breaking at: ", i)
                break
        if verbous:
            print("Best inliers: ", best_inliers)

        # remove detected ball
        border_coords = filter_detected_ball(best_model, r, border_coords)
        ball_centers.append(best_model)

    '''
    # prepare image for painting
    img=Image.fromarray(img)
    img.save('camera.png')
    d=ImageDraw.Draw(img)
    ball_coords=list(map(tuple,ball_coords))
    d.point(ball_coords,fill='red')
    border_mask, border_coords = get_border_mask(
        segmentation_mask, ball_coords, step)
    border_coords=list(map(tuple,border_coords))
    # print(border_coords)
    d.point(border_coords,fill='white') # only paints the remaining border

    
    for best_model in ball_centers:
        best_model=best_model
        bounding_box=[*(best_model-r),*(best_model+r)]
        d.ellipse(bounding_box,width=2,outline='blue')
    segmentation=Image.fromarray(255*segmentation_mask)
    border=Image.fromarray(255*border_mask)
    segmentation.save('segmentation.png')
    border.save('border.png')
    img.save('output.png')
    
    # print(best_inliers)
    # for best_model in ball_centers:
    #     best_model=best_model
    #     bounding_box=[*(best_model-r),*(best_model+r)]
    #     d.ellipse(bounding_box,width=2,outline='black')
    
    # im_to_show.show()'''
    return ball_centers
