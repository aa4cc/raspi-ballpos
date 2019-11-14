import numpy as np
from threading import RLock

'''
This file generates the center overlays during startup
these overlays are then moved appropriately by the GPU so that it does not affect performance
'''

lock = RLock()

def generate(size):
    try:
        overlay_size=size*32
        x = overlay_size
        y = overlay_size
        a = np.zeros((x,y,3), dtype=np.uint8)
        a[overlay_size//2, :, :] = 0xff
        a[:, overlay_size//2, :] = 0xff
        
        return a.tobytes(), a.shape[0:2], 'rgb', (int(a.shape[0]/2), int(a.shape[1]/2))
    except ValueError:
        raise ValueError('Argument "{}"is not valid overlay'.format(arg))


def move(overlay, position=(0,0), alpha=0):
    #print("moving")
    with lock:
        #print(type(overlay))
        overlay.window =(
            overlay.offset[0]-overlay.center[0]+int(position[0]*overlay.scale),
            overlay.offset[1]-overlay.center[1]+int(position[1]*overlay.scale),
            overlay.size[0],
            overlay.size[1])
        overlay.alpha = alpha

def new(camera, scale=1, size=2, offset=(0,0), alpha=0):
    buff, size, format, center = generate(size)
    o=camera.add_overlay(buff, layer=3, alpha=alpha, fullscreen=False, size=size, format=format, window=(0,0,size[0],size[1]))
    o.scale = scale
    o.center = center
    o.offset = offset
    o.size = size
    move(o)
    return o

def init(camera, count, **params):
    with lock:
        return [new(camera, **params) for _ in range(count)]

def destroy(camera):
    with lock:
        for o in list(camera.overlays):
            camera.remove_overlay(o)
