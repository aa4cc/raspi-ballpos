#!/usr/bin/python3

import time
import picamera
import pickle
import numpy as np
# import cv2
import socket
import struct
import click
import sys
import os.path
import logging
from fractions import Fraction

from interface import app as web_interface

from parameters import Parameters
from controller import Controller
from sharepos import SharedPosition
import detector
import processor

import lamp
import overlays

from screeninfo import get_monitors
# from imutils.video import FPS

# Global variables
logger = logging.getLogger(__name__)
params = Parameters("defaults.json")
NAN = float('nan')

scale = None
offset = None

wb_settings_filename='white_balance'

def get_sreen_resolution():
    try:
        print(get_monitors())
        m = get_monitors()[0]
        return m.width, m.height
    except (NotImplementedError, IndexError):
        if not all(params["screen_resolution"]):
            raise NotImplementedError(
                "Calculations for overlay not supported without X server, or screen resolution specified in config")
        return params['screen_resolution']


def change_wb(wb, camera):
    camera.awb_gains = wb
    with open(wb_settings_filename, 'wb') as settings_file:
        pickle.dump(wb, settings_file, pickle.HIGHEST_PROTOCOL)

def load_saved_wb(camera):
    with open(wb_settings_filename, 'rb') as settings_file:
        camera.awb_gains = pickle.load(settings_file)
    print("Old wb loaded!")

def setup(camera, params, processor):
    camera.resolution = params["resolution"]
    # Set the framerate appropriately; too fast and the image processors
    # will stall the image pipeline and crash the script
    camera.framerate = params['frame_rate']
    camera.shutter_speed = params['exposition_time']*1000
    camera.iso = params["iso"]
    camera.hflip = params["hflip"]
    camera.vflip = params["vflip"]

    if params["annotate"]:
        camera.annotate_foreground = picamera.Color(params["annotate"])
        camera.annotate_text = "Starting detection..."

    if params['preview']:
        # show live camera overlay
        try:
            screen_w, screen_h = get_sreen_resolution()
            print("Screen resolutiuon: {}x{} px".format(screen_w, screen_h))
            prev_w, prev_h = camera.resolution[0], camera.resolution[1]

            screen_r = screen_w/screen_h
            prev_r = prev_w/prev_h

            if screen_r > prev_r:
                h = int(screen_h)
                w = int(screen_h*prev_r)
            else:
                h = int(screen_w/prev_r)
                w = int(screen_w)
            global offset, scale
            offset = int((screen_w-w)/2), int((screen_h-h)/2)
            scale = w/prev_w

            t = (offset[0], offset[1], w, h)

            preview = camera.start_preview(fullscreen=False, window=t)

            if params["overlay"]:
                overlays.init(
                    camera,
                    processor.numberOfObjects(),
                    offset=offset,
                    size=params["overlay"],
                    alpha=params["overlay_alpha"],
                    scale=scale
                )

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
    if os.path.isfile(wb_settings_filename) and params['load_old_color_settings']:
        load_saved_wb(camera)
    elif all(params["white_balance"]):
        print("Manual white balance: ", params["white_balance"])
        change_wb(params["white_balance"],camera)
    else:
        print("Fixed auto white balance: {}".format(camera.awb_gains))
        change_wb(g,camera)

    print("Exposition time: {}".format(camera.exposure_speed/1000))
    print("camera.iso: {}".format(camera.iso))


def run(params, processor):
    with picamera.PiCamera() as camera:
        processor.recreate_detectors(params.detectors)
        shared_position = None
        fps = None
        recording = False
        try:
            shared_position = SharedPosition(processor.numberOfObjects())
            if params["neural-network"]:
                nn_controller = Controller(SharedPosition(
                    processor.numberOfObjects(), format='ff', key=3145915))
            setup(camera, params, processor)

            def position_callback(centers):
                # print(centers)
                # Write the measured position to the shared memory
                if shared_position:
                    shared_position.write_many(center if center else (
                        NAN, NAN, NAN) for center in centers)
                    # nn_controller.write()

                if params["preview"] and params["annotate"]:
                    camera.annotate_text = "Position:\n   {}".format(
                        "\n   ".join(map(str, centers)))
                if params["preview"] and params["overlay"]:
                    if len(centers) > len(camera.overlays):
                        while len(centers) > len(camera.overlays):
                            overlays.new(camera, offset=offset, size=params["overlay"],
                                         alpha=params["overlay_alpha"], scale=scale)
                    if len(centers) < len(camera.overlays):
                        for i in range(len(centers), len(camera.overlays)):
                            camera.remove_overlay(camera.overlays[i])
                    for o, center in zip(camera.overlays, centers):
                        if center:
                            overlays.move(
                                o, center, alpha=params["overlay_alpha"])
                        else:
                            overlays.move(o)
                # fps.update()
            processor.callback = position_callback

            if params['web_interface']:
                web_interface.camera = camera  # for access with webserver
                web_interface.processor = processor

            if params["video_record"]:
                try:
                    fName = os.path.join(params['img_path'], 'video.h264')
                    camera.start_recording(
                        fName, splitter_port=2, resize=params["resolution"])
                    recording = True
                    print("Recording video to:", fName)
                except FileNotFoundError:
                    print(
                        "Directory to store video recording ({}) not found or not accessible".format(fName))
                    return

            # fps = FPS().start()
            print("Starting capture")
            camera.capture_sequence(
                processor, use_video_port=True, format="rgb")

        finally:
            if fps:
                fps.stop()

            if params["video_record"] and recording:
                camera.stop_recording(splitter_port=2)
                print("Stop recording...")

            if params["preview"]:
                camera.stop_preview()

            if fps:
                print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        shared_position = None


def service(params, processor):
    if params['web_interface']:
        web_interface.start()

    while not processor.is_stopped():
        try:
            processor.stop_event.clear()
            run(params, processor)
        except KeyboardInterrupt:
            break


@ click.command()
@ click.option('--config-file', '-c', default="../config.json", type=click.Path(exists=True), help="Path to config file")
@ click.option('--verbose', '-v', count=True, default=False, help='Display time needed for processing of each frame and the measured positions.')
@ click.option('--preview', '-p', is_flag=True, default=False, help="Show preview on HDMI or display")
@ click.option('--video-record', is_flag=True, default=False, help="Record video")
@ click.option('--img-path', type=str, default='./img/', help="Path to store images and videos ideally ramdisk")
@ click.option('--interactive', '-i', is_flag=True, help="Start interactive Python console, to get realtime access to PiCamera object for testing purposes")
@ click.option('--multicore', is_flag=True, help="Start detectors in different processes to speedup detection")
@ click.option('--web-interface/--no-web-interface', is_flag=True, default=True, help="Enable/Disable web interface on port 5001 (default: enable)")
@ click.option('--load-old-color-settings', '-l', is_flag=True, default=False, help="Instead of loading color settings from JSON, values from last run will be used (for MultiColorDetector)")
@ click.option('--neural-network', '-n', is_flag=True, default=False, help='Whether to use neural network or not (not working yet)')
def main(**kwargs):
    # load json and kwargs and pass it to objects to use
    params.load(kwargs["config_file"])
    params.update(kwargs)
    detector.params = params
    processor.params = params
    web_interface.params = params

    camera = None
    proc = None
    mask = None

    print('FPS: {}'.format(params["frame_rate"]))

    if params['verbose']:
        print('Verbose: {}'.format(params['verbose']))

    if params['lamp_control'] is not None:
        lamp.pin = params['lamp_control']
        lamp.delay = params['lamp_delay']
        lamp.init(not params['lamp_manual'])

    if params['mask'] is not None:
        if os.path.isfile(params['mask']):
            mask = cv2.imread(params['mask'], 0)//255
        else:
            raise Exception('The mask with the given filename was not found!')

    if params["multicore"]:
        proc_class = processor.MultiCore
    else:
        proc_class = processor.SingleCore

    proc = proc_class(mask=mask)

    if not params["interactive"]:
        try:
            service(params, proc)
        except (KeyboardInterrupt, SystemExit):
            print('Yes, hold on; I am trying to kill myself!')
        finally:
            if proc is not None:
                proc.stop()

    else:
        import threading
        import code
        import readline
        import rlcompleter
        from pprint import pprint

        thread = threading.Thread(
            name="Camera thread", args=(params, proc), target=service)
        thread.start()

        from interface import app

        vars = globals()
        vars.update(locals())
        readline.set_completer(rlcompleter.Completer(vars).complete)
        readline.parse_and_bind("tab: complete")

        code.interact(local=vars)

        if proc is not None:
            proc.stop()
        thread.join(timeout=5)

    if params['lamp_control'] is not None:
        lamp.deinit()


if __name__ == '__main__':
    main(default_map=params.data)
