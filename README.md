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
1) cp raspi-ballpos/config.json_sample config.json
1) Edit your configuration using your favourite editor
1) cd raspi-ballpos
1) run posMeas.py -ivp and see the result

## Usage
Usage: posMeas.py [OPTIONS]

Options:

|Shortcut| Option | Type | Description|
|--|--|--|--|
| -n | \-\-num-frames | INTEGER |Total number of frames to process|
| -f | \-\-frame-rate | INTEGER | Number of frames per second to process|
| -e | \-\-exposition-time | INTEGER | Exposition time (shutter speed) in milliseconds.|
| -v | \-\-verbose| flag | Display time needed for processing of each frame and the measured positions.|
| -r | \-\-resolution | \<INTEGER INTEGER> | Image resolution|
| -p | \-\-preview |flag | Show preview on HDMI or display|
| | \-\-video-record | flag | Record video|
| | \-\-img-path | TEXT| Path to store images ideally ramdisk|
| -m | \-\-mask | PATH | Filename of mask to be applied on the captured images. The mask is assumed to be grayscale with values 255 for True and 0 for False.|
| -l | \-\-lamp-control | INTEGER | Pin for control external lamp|
| | \-\-lamp-delay | FLOAT |  Delay afrer lamp start to let the light warm up|
| | \-\-hflip / \-\-no-hflip | flag | Horizontal flip of image|
| | \-\-vflip / \-\-no-vflip |flag| Vertial flip of image|
| -a | \-\-annotate | TEXT | Color of position in preview|
| -o | \-\-overlay | TEXT | Enable overlay|
| |\-\-overlay-alpha | INTEGER | Overlay alpha|
| |\-\-iso | INTEGER | ISO for camera|
| -i | \-\-image-server | flag | Activate Image server |
| -w | \-\-white-balance | \<FLOAT FLOAT> | Camera white balance settings |
| | \-\-interactive| flag | Start interactive Python console, to get realtime access to PiCamera object for testing purposes
| |\-\-multicore| flag | Start detectors in different processes to speedup detection 
| |\-\-help |flag | Show this message and exit.|


> Written with [StackEdit](https://stackedit.io/).