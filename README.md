# RaspiBallPos

## Requirements
- Python3
- OpenCV3
Follow the instructions in [this guide](http://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/) to install OpenCV3 for python3.
We recommend not using `virtualenvs` on Raspberry Pi if this is the only project you'll be running on it (it is not necessary).
If your compilation fails, you might need to disable pre-compiled headers by adding the `-DENABLE_PRECOMPILED_HEADERS=OFF` flag when running cmake in the guide above.

## Installation
1) Enable Raspberry Camera module by running ```sudo raspi-config``` in case you haven't already done it.
1) Install the required Python modules by running ```sudo pip3 install picamera click flask matplotlib profilehooks screeninfo imutils RPi.GPIO```
1) Clone this repository by ```git clone https://github.com/aa4cc/raspi-ballpos.git```
1) Compile _sharemem_ module by running ```cd raspi-ballpos/sharemem; ./install```
1) Go back by ```cd ../..```
1) Run ```cp raspi-ballpos/config.json_sample config.json```
1) Edit your configuration using your favourite editor (do not forget to specify screen resolution, otherwise the detected object position will not be displayed correctly).
1) Run ```cd raspi-ballpos```
1) Run ```python3 vision.py -ivp``` and see the result.

### Automatic startup
1) In order to start the process automatically upon booting, copy ```vision.service``` into ```/etc/systemd/system/``` (run ```sudo cp vision.service /etc/systemd/system/vision.service```).
1) Open the file (```sudo nano /etc/systemd/system/vision.service```) and change *ExecStart* and *WorkingDirectory* paths to ```vision.py``` and ```raspi-ballpos``` respectively and optionally set flags for starting.
1) Use ```sudo systemctl ACTION vision``` with actions such as ```start```, ```stop```, ```enable```, ```disable``` to control the service.

## Usage
Usage: vision.py [OPTIONS]

Options:

|Shortcut| Option | Type | Description|
|--|--|--|--|
| -c | \-\-config-file | FILE PATH | Path to onfig file (default: ../config.json)|
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
- ```/image``` - Returns the latest PNG color image
- ```/image/DETECTOR/TYPE``` - Returns the latest PNG image from detector DETECTOR and TYPE chosen from ```image```, ```image_dwn```,```downsample```,```downsample_thrs```,```roi```,```roi_thrs```
- ```/imagesc/DETECTOR/TYPE``` - Same as ```/image/DETECTOR/TYPE``` but rendered by matplotlib imshow
- ```/wb``` - Page for white ballancing camera
- ```/lamp/on```, ```/lamp/off``` - Turn lamp on and off respectively
- ```/restart``` - Restart image detection subsystem