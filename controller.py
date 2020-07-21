import numpy as np
from pathlib import Path
import scipy.io as sio
class Controller:
    def __init__(self, force_share_mem, path=None):
        if path is not None:
            self.path=path
        else:
            path=Path(__file__).parent.absolute().parent.joinpath('control/8x8/km8x8.mat')
        self.NAN = float('nan')
        self.km=None
        self.share_mem=force_share_mem
        if path.exists():
            self.km = np.array(sio.loadmat(str(path))['km8x8'])
        else: 
            print("Expected calibration matrix at {} but didn't find it (it can be generated using calibrate.m). Aborting...".format(path))
            quit()

    def homography_transform(self,centers):
        transformed=[]
        for center in centers:
            if center is None:
                transformed.append(None)
            else:
                center=np.array(center)
                center[2]=1 # add homogeneous coord instead of unused theta
                q=self.km@center
                p=q[2]
                transformed.append(q[0:2]/p)
        return transformed
    
    def write(self):
        self.share_mem.write_many((0,0),(1,1),(2,2),(3,3),(4,4),(5,5))#(center if center else (NAN, NAN, NAN) for center in centers)

#c=Controller()
#print(c.homography_transform([None, (0,0,0),(10,10,10)]))