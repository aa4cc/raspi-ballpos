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
from fractions import Fraction

from interface import app

from parameters import Parameters

from sharepos import SharedPosition
import detector
import processor

from screeninfo import get_monitors
from imutils.video import FPS

# TODO
# Order of addding color components

# Global variables
params = Parameters("../config.json")
NAN = float('nan')

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

def move_overlay(overlay, position, active=True):
    overlay.window =(
        overlay.offset[0]-overlay.center[0]+int(position[0]*overlay.scale),
        overlay.offset[1]-overlay.center[1]+int(position[1]*overlay.scale),
        overlay.size[0],
        overlay.size[1])
    overlay.alpha=params["overlay_alpha"] if active else 0


def parameter_checks():
    # Ball mass - ball is approximately 40 px in diameter in the image hence the mass should be somewhere around pi*20^2=1256.
    # The values are multiplied by 255 because the the pixels in the binary image have values 0 and 255 (weird, isn't it?).
    print('Number of frames: {num_frames}'.format(**params))
    print('FPS: {frame_rate}'.format(**params))

    if params['verbose']:
        print('Verbose: {}'.format(params['verbose']))

def pre_camera_tasks():
    if params['web_interface']:
        app.start()

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

def get_sreen_resolution():
    try:
        m = get_monitors()[0]
        return m.width, m.height
    except NotImplementedError:
        if not all(params["screen_resolution"]):
            raise NotImplementedError("Calculations for overlay not supported without X server, or screen resolution specified in config")
        return params['screen_resolution']

def camera_setup(camera, processor):
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
            screen_w, screen_h = get_sreen_resolution()
            print("Screen resolutiuon: {}x{} px".format(screen_w, screen_h))
            prev_w, prev_h = camera.resolution[0], camera.resolution[1]

            screen_r = screen_w/screen_h
            prev_r = prev_w/prev_h

            if screen_r > prev_r:
                h = screen_h
                w = int(screen_h*prev_r)
            else:
                h = screen_w/prev_r
                w = int(screen_w)

            offset = int((screen_w-w)/2), int((screen_h-h)/2)
            scale = w/prev_w

            preview = camera.start_preview(fullscreen=False, window=(offset[0],offset[1],w,h))

            if params["overlay"]:
                for detector in processor.detectors:
                    buff, size, format, center = gen_overlay(params["overlay"])
                    o=camera.add_overlay(buff, layer=3, alpha=params["overlay_alpha"], fullscreen=False, size=size, format=format, window=(0,0,size[0],size[1]))
                    o.scale = scale
                    o.center = center
                    o.offset = offset
                    o.size = size
                    move_overlay(o, (0,0))
                    o.name = detector.name

        except NotImplementedError as e:
            print(e)
            params["overlay"] = None
            preview = camera.start_preview()

    # Let the camera warm up
    time.sleep(2)

    # Now fix the values
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'

    if all(params["white_balance"]):
        print("Manual white balance: ", params["white_balance"])
        camera.awb_gains = params["white_balance"]
    else:
        print("Fixed auto white balance: {}".format(camera.awb_gains))
        camera.awb_gains = g

    print("Exposition time: {}".format(camera.exposure_speed/1000))
    print("camera.iso: {}".format(camera.iso))

@click.command()
@click.option('--num-frames', '-n', default=0, help='Total number of frames to process, then exit')
@click.option('--frame-rate', '-f', default=10, help='Number of frames per second to process')
@click.option('--exposition-time', '-e', default=10, help='Exposition time (shutter speed) in milliseconds.')
@click.option('--verbose', '-v', count=True, default=False, help='Display time needed for processing of each frame and the measured positions.')
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
@click.option('--image-server', '-i', is_flag=True, help="Activate Image server")
@click.option('--white-balance', '-w', type=(float, float), default=(None, None), help="Camera white balance settings")
@click.option('--interactive', is_flag=True, help="Start interactive Python console, to get realtime access to PiCamera object for testing purposes")
@click.option('--multicore', is_flag=True, help="Start detectors in different processes to speedup detection")
@click.option('--screen-resolution', type=(int, int), default=(None, None))
@click.option('--web-interface', is_flag=True)
def main(**kwargs):
    global params
    params.update(kwargs)
    detector.params = params
    processor.params = params

    parameter_checks()
    pre_camera_tasks()
    camera = None
    proc = None

    shared_position = SharedPosition(len(params.detectors))
    try:
        fps = FPS().start()
        with picamera.PiCamera() as camera:
            app.camera = camera #for access with webserver
            
            def position_callback(centers):
                # Write the measured position to the shared memory
                shared_position.write_many(center if center else (NAN, NAN, NAN) for center in centers)

                if params["preview"] and params["annotate"]:
                    camera.annotate_text = "Position:\n   {}".format("\n   ".join(map(str, centers)))
                if params["preview"] and params["overlay"]:
                    for overlay, center in zip(camera.overlays, centers):
                        if center:
                            move_overlay(overlay, center)
                        else:
                            move_overlay(overlay, (0,0), active=False)
                fps.update()

            if params["multicore"]:
                proc_class = processor.MultiCore
            else:
                proc_class = processor.SingleCore

            mask = None
            if params['mask'] is not None:
                if os.path.isfile(params['mask']):
                    mask = cv2.imread(params['mask'], 0)//255
                else:
                    raise Exception('The mask with the given filename was not found!')

            proc = proc_class(params.detectors,position_callback, mask=mask)

            camera_setup(camera, proc)

            if params["video_record"]:
                camera.start_recording('{}video.h264'.format(params['img_path']), splitter_port=2, resize=params["resolution"])

            fps.start()
            print("Starting capture")
            if not params["interactive"]:
                camera.capture_sequence(proc, use_video_port=True, format="rgb")
            else:
                import threading
                import code
                import readline
                import rlcompleter
                threading.Thread(target=camera.capture_sequence, args=(proc,), kwargs={"use_video_port": True, "format":"rgb"}).start();

                vars = globals()
                vars.update(locals())
                readline.set_completer(rlcompleter.Completer(vars).complete)
                readline.parse_and_bind("tab: complete")

                code.interact(local=vars)

            if params["video_record"]:
                camera.stop_recording(splitter_port=2)

            if params["preview"]:
                camera.stop_preview()

    except (KeyboardInterrupt, SystemExit):
        print('Yes, hold on; I am trying to kill myself!')
    finally:
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # Shut down the processors in an orderly fashion
        if camera is not None:
            camera.close()

        if proc is not None:
            proc.stop()

        if params['lamp_control'] is not None and GPIO:
            GPIO.output(params['lamp_control'], False);
            GPIO.cleanup()

if __name__=='__main__':
    main(default_map=params.data)
