import ctypes
from ctypes import c_float, c_uint8, c_int, POINTER, pointer, c_double, byref, c_bool, c_size_t
import numpy as np
from numpy.ctypeslib import ndpointer


class Ball_t(ctypes.Structure):
    # this class is used to interface with RANSAC C
    _fields_ = [("h_min", c_double),
                ("h_max", c_double),
                ("sat_min", c_double),
                ("val_min", c_double)]


class Coord_t(ctypes.Structure):
    # this class is used to interface with RANSAC C
    _fields_ = [("x", c_float),
                ("y", c_float)]


class Coords_t(ctypes.Structure):
    # this class is used to interface with RANSAC C
    _fields_ = [("length", c_size_t),
                ("allocated", c_size_t),
                ("coords", POINTER(Coord_t))]


class IntCoord_t(ctypes.Structure):
    # this class is used to interface with RANSAC C
    _fields_ = [("x", c_int),
                ("y", c_int)]


class IntCoords_t(ctypes.Structure):
    # this class is used to interface with RANSAC C
    _fields_ = [("length", c_size_t),
                ("allocated", c_size_t),
                ("coords", POINTER(IntCoord_t))]


class Indexes_t(ctypes.Structure):
    # this class is used to interface with RANSAC C
    _fields_ = [("length", c_size_t),
                ("allocated", c_size_t),
                ("indexes", POINTER(c_int))]


def wrapped_ndptr(*args, **kwargs):
    base = ndpointer(*args, **kwargs)

    def from_param(cls, obj):
        if obj is None:
            return obj
        return base.from_param(obj)
    return type(base.__name__, (base,), {'from_param': classmethod(from_param)})


def detector_funcs():
    so_file = './ransac_detector.so'
    funcs = ctypes.cdll.LoadLibrary(so_file)

    funcs.init_table.argtypes = [
        ndpointer(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS'),
        POINTER(Ball_t),
        c_int
    ]

    funcs.get_ball_pixels.argtypes = [
        ndpointer(dtype=np.uint8, ndim=3,
                  flags='C_CONTIGUOUS'),  # img
        c_int,  # width
        c_int,  # height
        c_int,  # number of colors
        ndpointer(dtype=np.uint8, ndim=3,
                  flags='C_CONTIGUOUS'),  # table
        c_int,  # step
        ndpointer(dtype=np.uint8, ndim=2,
                  flags='C_CONTIGUOUS'),  # mask (empty)
        POINTER(IntCoords_t),  # ball_pixels
    ]

    funcs.get_segmentation_mask.argtypes = [
        ndpointer(dtype=np.uint8, ndim=3,
                  flags='C_CONTIGUOUS'),  # img
        c_int,  # width
        c_int,  # height
        ndpointer(dtype=np.uint8, ndim=2,
                  flags='C_CONTIGUOUS'),  # mask
        POINTER(Ball_t),
    ]

    funcs.get_border.argtypes = [
        ndpointer(dtype=np.uint8, ndim=2,
                  flags='C_CONTIGUOUS'),  # seg_mask (full)
        c_int,  # width
        c_int,  # height
        POINTER(IntCoords_t),  # ball_pixel_coords
        POINTER(Coords_t),  # previous_positions
        c_int,  # step
        c_float,  # max_dx2
        ndpointer(dtype=np.uint8, ndim=2,
                  flags='C_CONTIGUOUS'),  # border_mask (empty)
        ndpointer(dtype=np.uint8, ndim=2,
                  flags='C_CONTIGUOUS'),  # group (empty)
        POINTER(POINTER(Indexes_t)),  # groups
        POINTER(IntCoords_t),  # border
    ]

    funcs.ransac.argtypes = [
        POINTER(IntCoords_t),  # border_coords
        c_float,  # r
        c_float,  # min_dst
        c_float,  # max_dst
        c_int,  # max_iter
        c_int,  # confidence_thrs
        c_bool,  # verbose
        POINTER(Coord_t),  # best_model_ret
    ]

    funcs.find_modeled_pixels.argtypes = [
        POINTER(Coord_t),  # model
        c_float,  # max_dx2
        c_float,  # min_dist
        c_float,  # max_dist
        POINTER(IntCoords_t),  # border
        POINTER(Indexes_t)  # modeled_indexes
    ]

    funcs.remove_pixels.argtypes = [
        POINTER(IntCoords_t),  # border_coords
        POINTER(Indexes_t),  # group_indexes
        POINTER(Indexes_t),  # modeled_indexes
        c_bool  # only group
    ]

    funcs.detect_balls.argtypes = [
        ndpointer(dtype=np.uint8, ndim=3,
                  flags='C_CONTIGUOUS'),  # table
        ndpointer(dtype=np.uint8, ndim=3,
                  flags='C_CONTIGUOUS'),  # img
        c_int,  # width
        c_int,  # height
        c_int,  # step
        POINTER(Coords_t),  # previous_positions
        POINTER(c_int),  # ball colors
        c_float,  # max_dx
        c_float,  # r
        c_float,  # min_dst
        c_float,  # max_dst
        c_int,  # max_iter
        c_int,  # confidence_thrs
        c_bool,  # verbose
        POINTER(Coord_t),  # best_model_ransac
        POINTER(Coord_t),  # best_model_coope
    ]

    return funcs
