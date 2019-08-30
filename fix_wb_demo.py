#!/usr/bin/python3

import picamera
import numpy as np
import time

def main():
	with picamera.PiCamera() as camera:
		print(camera.revision)
		camera.exposure_mode = 'off'
		camera.resolution = (480,480)
		camera.awb_mode = 'off'
		camera.awb_gains = (1.05, 1.95)
		camera.framerate = 50      
		camera.shutter_speed = 10000
		camera.iso = 200
		camera.hflip = False
		camera.vflip = False


		camera.start_preview()
		camera.start_recording("wb_video.mjpeg", format='mjpeg')
		input("Press enter to take first picture")
		camera.capture('wb_image1_use_video_port_false.jpg', use_video_port=False)
		time.sleep(0.5)
		camera.capture('wb_image1_use_video_port_true.jpg', use_video_port=True)
		input("Press enter to take second picture")
		camera.capture('wb_image2_use_video_port_false.jpg', use_video_port=False)
		time.sleep(0.5)
		camera.capture('wb_image2_use_video_port_true.jpg', use_video_port=True)
		input("Press enter to terminate")
		camera.stop_recording()
		camera.stop_preview()


if __name__=='__main__':
    main()