#!/usr/bin/python3

import time
import picamera
import numpy as np
import cv2
import socket, struct
import click
import sys
import os.path
import logging

try:
    # Add the parent directory to the path
    import os.path, sys
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    from config import config, detectors
except ImportError as ex:
    logging.exception(ex)
    sys.exit('No default configuration file found')

import ballposition
import detector
import processor

from screeninfo import get_monitors
from imutils.video import FPS

# TODO
# Order of addding color components

# Global variables
params = {}

def gen_overlay(arg):
    try:
        size = int(arg)
        x = size*32
        y = size*32
        a = np.zeros((x,y,3), dtype=np.uint8)
        a[size*16, :, :] = 0xff
        a[:, size*16, :] = 0xff
        return a.tobytes(), a.shape[0:2], 'rgb', (int(a.shape[0]/2), int(a.shape[1]/2))
    except ValueError:
        raise ValueError('Argument "{}"is not valid overlay'.format(arg))

def parameter_checks():
    # Ball mass - ball is approximately 40 px in diameter in the image hence the mass should be somewhere around pi*20^2=1256.
    # The values are multiplied by 255 because the the pixels in the binary image have values 0 and 255 (weird, isn't it?).
    print('Number of frames: {num_frames}'.format(**params))
    print('FPS: {frame_rate}'.format(**params))

    if params['verbose']:
        print('Verbose')
    if params['debug']:
        print('Debug {}'.format(params["debug"]))

def pre_camera_tasks():
    if params['lamp_control'] is not None:
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            print("Starting lamp....     ", end="", flush=True)
            GPIO.setup(params['lamp_control'], GPIO.OUT)
            GPIO.output(params['lamp_control'], True);
            # Let the light warm up
            time.sleep(params['lamp_delay'])
            print("OK")
        except RuntimeError:
            print("Error importing RPi.GPIO!  This is probably because you need superuser privileges. You can achieve this by using 'sudo' to run your script")
        except ImportError:
            print("Libratry RPi.GPIO not found, light controll not possible! You can install it using 'sudo pip3 install rpi.gpio' to install library")

def camera_setup(camera):
    camera.resolution = params["resolution"]
    # Set the framerate appropriately; too fast and the image processors
    # will stall the image pipeline and crash the script
    camera.framerate = params['frame_rate']        
    camera.shutter_speed = params['exposition_time']*1000
    camera.iso = params["iso"];
    camera.hflip = params["hflip"]
    camera.vflip = params["vflip"]

    if params["annotate"]:
        camera.annotate_foreground = picamera.Color(params["annotate"])
        camera.annotate_text = "Starting detection..."

    if params['preview']:
        try:
            screen_w, screen_h = get_monitors()[0].width, get_monitors()[0].height
            prev_w, prev_h = camera.resolution[0], camera.resolution[1]

            screen_r = screen_w/screen_h
            prev_r = prev_w/prev_h

            if screen_r > prev_r:
                h = screen_h
                w = int(screen_h*prev_r)
            else:
                h = screen_w/prev_r
                w = int(screen_w)

            offset_x = int((screen_w-w)/2)
            offset_y = int((screen_h-h)/2)
            scale = w/prev_w

            preview = camera.start_preview(fullscreen=False, window=(offset_x,offset_y,w,h))

            if params["overlay"]:
                buff, size, format, center = gen_overlay(params["overlay"])
                o=camera.add_overlay(buff, layer=3, alpha=params["overlay_alpha"], fullscreen=False, size=size, format=format, window=(0,0,size[0],size[1]))

                def move_overlay(x,y, center_x=center[0], center_y=center[1]):
                    o.window = (offset_x-center_x+int(x*scale), offset_y-center[1]+int(y*scale), size[0], size[1])

                o.move = move_overlay
                o.move(0,0)
        except NotImplementedError:
            print("Calculations for overlay not supported without X server (Cannot get monitor resolution)")
            params["overlay"] = None
            preview = camera.start_preview()

    # Let the camera warm up
    time.sleep(2)

    print("Exposition time: {}".format(camera.exposure_speed/1000))
    print("camera.awb_gains: {}".format(camera.awb_gains))
    print("camera.iso: {}".format(camera.iso))

    # Now fix the values
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g

@click.command()
@click.option('--num-frames', '-n', default=0, help='Total number of frames to process')
@click.option('--frame-rate', '-f', default=10, help='Number of frames per second to process')
@click.option('--exposition-time', '-e', default=10, help='Exposition time (shutter speed) in milliseconds.')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Display time needed for processing of each frame and the measured position.')
@click.option('--debug', '-d', count=True, default=False, help='Save masks and ROIs together with the identified position of the ball.')
@click.option('--resolution', '-r', type=(int, int), default=(640,480), help='Image resolution')
@click.option('--preview', '-p', is_flag=True, default=False, help="Show preview on HDMI or display")
@click.option('--video-record', is_flag=True, default=False, help="Record video")
@click.option('--img-path', type=str, default='./img/', help='Path to store images, ideally ramdisk')
@click.option('--mask', '-m',type=(str), default=None, help="Filename of mask to be applied on the captured images. The mask is assumed to be grayscale with values 255 for True and 0 for False.")
@click.option('--lamp-control', '-l', type=int, default=None, help="Pin for control external lamp")
@click.option('--lamp-delay', type=float, default=2, help="Pin for control external lamp")
@click.option('--hflip/--no-hflip', is_flag=True, default=False, help="Horizontal flip of image")
@click.option('--vflip/--no-vflip', is_flag=True, default=False, help="Vertial flip of image")
@click.option('--annotate', '-a', default=None, help="Color of position in preview")
@click.option('--overlay', '-o', default=None, help='Enable overlay')
@click.option('--overlay-alpha', default=50, help='Overlay alpha')
@click.option('--iso', default=200, help="ISO for camera")
def main(**kwargs):
    global params
    params = kwargs
    detector.params = params
    processor.params = params

    parameter_checks()
    pre_camera_tasks()
    camera = None
    try:
        with picamera.PiCamera() as camera:
            def position_callback(center):
                # Write the measured position to the shred memory
                if center[0]:
                    ballposition.write(center[0])
                else:
                    ballposition.write((params["resolution"][0]+1, params["resolution"][1]+1))
                if params["annotate"]:
                    camera.annotate_text = "Position: {}".format(center)
                if params["overlay"] and center:
                    camera.overlays[0].move(*center[0])
                fps.update()

            proc = processor.Processor(detectors,position_callback)

            camera_setup(camera)

            if params['mask'] is not None:
                if os.path.isfile(params['mask']):
                    proc.mask = cv2.imread(params['mask'], 0)//255
                    proc.mask_dwn = processor.mask[::params["downsample"], ::params["downsample"]]
                else:
                    raise Exception('The mask with the given filename was not found!')

            if params["video_record"]:
                camera.start_recording('{}video.h264'.format(params['img_path']), splitter_port=2, resize=params["resolution"])

            fps = FPS().start()
            print("Starting capture")
            camera.capture_sequence(proc, use_video_port=True, format="rgb")
            fps.stop()

            if params["video_record"]:
                camera.stop_recording(splitter_port=2)

            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

            if params["preview"]:
                camera.stop_preview()

    except (KeyboardInterrupt, SystemExit):
        print('Yes, hold on; I am trying to kill myself!')

    finally:
        # Shut down the processors in an orderly fashion
        if camera is not None:
            camera.close()

        if params['lamp_control'] is not None and GPIO:
            GPIO.output(params['lamp_control'], False);
            GPIO.cleanup()

if __name__=='__main__':
    main(default_map=config)
