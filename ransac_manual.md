# Ransac manual
This is the manual for the RANSAC algorithm that is currently used to detect balls.

## High-level overview
The RansacDetector (specified in ```detector.py```) uses a C library (source in ```ransac/```).
First, an RGB-to-ball color table is initialized. For every RGB color, the value at that index specifies the ball index. This means that conversion of the image to HSV at every iteration isn't necessary.

In the loop, the library is called with the image from the camera as a parameter. This image is then segmented using the table, also outputing a list of coordinates for each of the ball colors. Then, for each of the colors, border is found (again as a list of coords).
The algorithm also tries to group the pixels based on previous positions.

Ransac then picks two border coords at random and uses them to fit (two) circles and calculates how many border pixels fit the model (more on that below).
This is done until either a preset number of pixels fit the model or the maximum number of iterations is reached.
Least squares fit is then used to improve the precision and the pixels that fit this final model are removed from the border.

## Web view
To visualize what the algorithm is detecting, go to ```http://[IP_ADDRESS]:5001/ransac```. It is possible to view different overlays of what the program detects (e.g. background, border, fit, etc.).

### Settings
The webpage is also used to control the behaviour of the algorithm. 
By pressing Enter, the settings are saved and the webpage should be updated.
* ```ball radius```: the radius of the circle being fitted
* ```maximum iterations```: the maximum number of iterations of ransac
* ```confidence threshold```: the amount of pixels necessary to stop ransac before reaching maximum iterations - note that the algorithm won't find the ball, if the amount of border pixels found is lower than confidence threshold (otherwise, any stray pixel might cause a wrong fit if a ball were missing)
* ```downsampling```: if the algorithm runs too slow, it is possible to sample only every k-th pixel (should be kept as high as possible for maximum accuracy)
* ```tolerance coeff```: used to set the annulus used to decide, whether a pixel fits the ransac model; the value is computed by multiplying it by the radius
    * note that pixels removed are currently hardcoded to ```[0.9*r, 1.1*r]``` of the LSQ fit
* ```maximum expected movement per frame```: this value is related to how the border coords are groupped - if the coord is closer than this value to previous position (in pixels), it is considered a plausible candidate for new position; keeping this too high will slow down the algorithm (more pixels taken into account), while too low will mean that the ball will have to be found among all border coords of the same color (thus again slowing down the detection)
* ```ball color amounts```: these values specify how many balls are looked up, with backgrounds specifying what color the field controls. It is there where additional colors may be added (by pressing '+' and then changing color) or removed (simply by setting 0 for some of the colors).

Furthermore, ```http://[IP_ADDRESS]:5001/ball_colors``` is used to set ball colors.