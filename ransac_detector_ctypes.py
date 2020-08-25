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
        ndpointer(dtype=np.uint8, ndim=3,
                  flags='C_CONTIGUOUS'),  # table
        c_int,  # step
        ndpointer(dtype=np.uint8, ndim=2,
                  flags='C_CONTIGUOUS'),  # mask (empty)
        POINTER(POINTER(c_int))  # ball coords length (len([x1,y1,x2,y2])=2)
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

    # length of the ball coords for parsing
    funcs.get_segmentation_mask.restype = c_size_t

    funcs.get_border_coords.argtypes = [
        ndpointer(dtype=np.uint8, ndim=2,
                  flags='C_CONTIGUOUS'),  # seg_mask (full)
        c_int,  # width
        c_int,  # height
        POINTER(c_int),  # ball_pixel_coords
        c_size_t,  # ball_pixel_coords_l
        POINTER(Coord_t),  # previous_positions
        c_size_t,  # previous_pos_l
        c_int,  # step
        c_float,  # max_dx
        ndpointer(dtype=np.uint8, ndim=2,
                  flags='C_CONTIGUOUS'),  # border_mask (empty)
        ndpointer(dtype=np.uint8, ndim=2,
                  flags='C_CONTIGUOUS'),  # group (empty)
        POINTER(POINTER(POINTER(c_int))),  # group_index_ret
        POINTER(POINTER(c_size_t)),  # group_index_ls_ret
        POINTER(POINTER(c_int)),  # border_coords_ret
    ]

    # length of the border coords for parsing
    funcs.get_border_coords.restype = c_size_t

    funcs.ransac.argtypes = [
        POINTER(Coord_t),  # border_coords
        c_size_t,  # border_coords_l
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
        POINTER(Coord_t),  # previous_center
        c_float,  # max_dx
        c_float,  # min_dst
        c_float,  # max_dst
        POINTER(Coord_t),  # border_coords
        c_size_t,  # border_coords_l
        POINTER(c_int)  # modeled_indexes
    ]

    funcs.find_modeled_pixels.restype = c_size_t

    funcs.remove_pixels.argtypes = [
        POINTER(Coord_t),  # border_coords
        c_size_t,  # l
        POINTER(c_int),  # group_indexes
        c_size_t,  # l
        POINTER(c_int),  # modeled_indexes
        c_size_t  # l
    ]

    funcs.detect_balls.argtypes = [
        ndpointer(dtype=np.uint8, ndim=3,
                  flags='C_CONTIGUOUS'),  # table
        ndpointer(dtype=np.uint8, ndim=3,
                  flags='C_CONTIGUOUS'),  # img
        c_int,  # width
        c_int,  # height
        c_int,  # step
        POINTER(Coord_t),  # previous_positions
        c_size_t,  # previous_pos_l
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
