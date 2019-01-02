# RaspiBallPos

## Requirements
- Python3
- OpenCV3
Follow the instruction in [this guide](http://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/) and install it for python3
We recomend you not to use virtuakenvs on Raspberry Pi it this is the only one project

## Instalation
1) Enable Raspberry Camera module by running ```sudo raspi-config```
1) Install required Python modules by running ```sudo pip3 install picamera click flask matplotlib profilehooks screeninfo imutils RPi.GPIO```
1) git clone https://github.com/aa4cc/raspi-ballpos.git
1) Compile _sharemem_ module by running ```sudo ./sharemem/install```
1) ```cp raspi-ballpos/config.json_sample config.json```
1) Edit your configuration using your favourite editor
1) ```cd raspi-ballpos```
1) run ```vision.py -ivp``` and see the result

## Usage
Usage: vision.py [OPTIONS]

Options:

|Shortcut| Option | Type | Description|
|--|--|--|--|
| -c | \-\-config-file | FILE PATH | Path to onfig file (default: ../config.json)|
| -f | \-\-frame-rate | INTEGER | Number of frames per second to process|
| -e | \-\-exposition-time | INTEGER | Exposition time (shutter speed) in milliseconds.|
| -v | \-\-verbose| flag | Display time needed for processing of each frame and the measured positions.|
| -p | \-\-preview |flag | Show preview on HDMI or display|
| | \-\-video-record | flag | Record video|
| | \-\-img-path | TEXT| Path to store images and videos ideally ramdisk|
| | \-\-interactive| flag | Start interactive Python console, to get realtime access to PiCamera object for testing purposes |
| |\-\-multicore| flag | Start detectors in different processes to speedup detection |
| |\-\-web-interface/--no-web-interface | flag | Enable/Disable web interface on port 5001 (default: enable) |
| |\-\-help |flag | Show this message and exit.|

## Web interface

If web interface is enabled, it can be accessed by IP address or hostname on port 5001. All code for webinterface is based in [interface.py](https://github.com/aa4cc/raspi-ballpos/blob/master/interface.py) file.

Example: ```HTTP://yourhostname.com:5001```

URL endpoints:

- ```/``` Summary page - Print all used settings and live images for all decoders
- ```/detector/NAME``` - Print settings and live images for deetector named NAME
- ```/centers``` - Returns JSON with positions of all objects
- ```/config``` - Returns live config in JSON which can be used to replicates settings on another instance in config.json
- ```/config``` - POST request loads new config form request into live config and restarts detection subsystem
- ```/config/loadfile``` - POST request loads new config from local filesystem by given filename into live config and restarts detection subsystem
- ```/detector/NAME/threshold/VALUE``` - Sets threshold of detector NAME to VALUE
- ```/image``` - Returns lastest PNG color image
- ```/image/DETECTOR/TYPE``` - Returns lastest PNG image from detector DETECTOR ant TYPE choosen from ```image```, ```image_dwn```,```downsample```,```downsample_thrs```,```roi```,```roi_thrs```
- ```/imagesc/DETECTOR/TYPE``` - same as ```/image/DETECTOR/TYPE``` but rendered by matplotlib imshow
- ```/wb``` -page for white ballancing camera
- ```/lamp/on```, ```/lamp/off``` - Turn lamp on and off respectively
- ```/restart``` - restart image detection subsystem


> Written with [StackEdit](https://stackedit.io/).