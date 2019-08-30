#include "math.h"
#include "numpy/arrayobject.h"
#include <Python.h>

#include <stdint.h>
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

// define structs
typedef struct {
  uint8_t r;
  uint8_t g;
  uint8_t b;
} Color_t;

typedef struct {
  double h_min;   // hue as an angle between 0 and 359
  double h_max;   // hue tolerance (+-) as an angle between 0 and 359
  double sat_min; // a fraction between 0 and 1
  double val_min; // a fraction between 0 and 1
} Ball_t;

typedef struct {
  double r; // a fraction between 0 and 1
  double g; // a fraction between 0 and 1
  double b; // a fraction between 0 and 1
} rgb_t;

typedef struct {
  double h; // angle in degrees
  double s; // a fraction between 0 and 1
  double v; // a fraction between 0 and 1
} hsv_t;

// define constants
#define R 0
#define G 1
#define B 2

#define X 1
#define Y 0

#define NONE 255

// any other resolution seems really bad
#define COLOR_RESOLUTION 256

#define MIN_VALUE 0.5

// define macros
#define min(a, b) ((a < b) ? a : b)
#define max(a, b) ((a > b) ? a : b)

#define pixel(im, x, y) (im + (y * sx) + x)

// define global variables
uint8_t
    rgb_to_balls_map[COLOR_RESOLUTION * COLOR_RESOLUTION * COLOR_RESOLUTION];
uint8_t map_initiated = 0;
// this is here so that you can't accidently ask from Python for more colors
// than we have values for
uint8_t number_of_colors = 0;

// translates an RGB value to table index of said value
int index_from_rgb(Color_t pixel) {
  return pixel.r / (256 / COLOR_RESOLUTION) * COLOR_RESOLUTION *
             COLOR_RESOLUTION +
         pixel.g / (256 / COLOR_RESOLUTION) * COLOR_RESOLUTION +
         pixel.b / (256 / COLOR_RESOLUTION);
}

// converts RGB color to HSV
hsv_t rgb2hsv(rgb_t in) {
  hsv_t out;
  double min, max, delta;

  min = in.r < in.g ? in.r : in.g;
  min = min < in.b ? min : in.b;

  max = in.r > in.g ? in.r : in.g;
  max = max > in.b ? max : in.b;

  out.v = max; // v
  delta = max - min;
  if (delta < 0.00001) {
    out.s = 0;
    out.h = 0; // undefined, maybe nan?
    return out;
  }
  if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
    out.s = (delta / max); // s
  } else {
    // if max is 0, then r = g = b = 0
    // s = 0, h is undefined
    out.s = 0.0;
    out.h = NAN; // its now undefined
    return out;
  }
  if (in.r >= max)                 // > is bogus, just keeps compiler happy
    out.h = (in.g - in.b) / delta; // between yellow & magenta
  else if (in.g >= max)
    out.h = 2.0 + (in.b - in.r) / delta; // between cyan & yellow
  else
    out.h = 4.0 + (in.r - in.g) / delta; // between magenta & cyan

  out.h *= 60.0; // degrees

  if (out.h < 0.0)
    out.h += 360.0;

  return out;
}

// generates a table of which RGB values belong to which ball, while balls are
// defined in HSV
void init_table(Ball_t *ball_params, int param_len) {
  for (int r = 0; r < COLOR_RESOLUTION; r++) {
    for (int g = 0; g < COLOR_RESOLUTION; g++) {
      for (int b = 0; b < COLOR_RESOLUTION; b++) {
        // convert RGB to a double between 0 and 1, only take every i.e. fourth
        // value
        double rd = 256 / COLOR_RESOLUTION * (double)r / 256;
        double gd = 256 / COLOR_RESOLUTION * (double)g / 256;
        double bd = 256 / COLOR_RESOLUTION * (double)b / 256;
        rgb_t pixelrgb = {rd, gd, bd};
        hsv_t pixelhsv = rgb2hsv(pixelrgb);
        Color_t pixel = {r, g, b};
        // for each ball, check if pixel color is within the limits and if so,
        // set its color in the table
        uint8_t found = 0;
        for (int c = 0; c < param_len; c++) {
          Ball_t ball = ball_params[c];
          if (((ball.h_min < pixelhsv.h && ball.h_max > pixelhsv.h) ||
               (ball.h_min < pixelhsv.h + 360 &&
                ball.h_max > pixelhsv.h + 360)) &&
              ball.sat_min < pixelhsv.s && pixelhsv.v > ball.val_min) {
            /*if (r == 130 && g == 120)
              printf("Color R: %d, G: %d, B:%d classified as %d\n", r, g, b,
              c);*/
            rgb_to_balls_map[index_from_rgb(pixel)] = c;
            found = 1;
            break;
          }
        }
        if (!found) {
          rgb_to_balls_map[r * COLOR_RESOLUTION * COLOR_RESOLUTION +
                           g * COLOR_RESOLUTION + b] = NONE;
        }
      }
    }
  }
  // set variables for other functions to know that table was initialized
  map_initiated = 1;
  number_of_colors = param_len;
  printf("C program received %d colors:\n", number_of_colors);
  for (int i = 0; i < number_of_colors; ++i) {
    printf("Color %d: H_MIN: %f, H_MAX:%f, SAT_MIN:%f, VAL_MIN: %f\n", i,
           ball_params[i].h_min, ball_params[i].h_max, ball_params[i].sat_min,
           ball_params[i].val_min);
  }
}

// finds centers of mass (and rotation of triangles) of all balls in the picture
static int find_location(float result[], Color_t image[], uint8_t image_thrs[],
                         int sx, int sy, int min_size, int max_size,
                         int compute_orientation, float orientation_offset,
                         int downsample, int start_x, int start_y, int end_x,
                         int end_y, uint8_t mask[], uint8_t only_ball_color) {

  // this struct is used to find the center of gravity of the balls (and
  // rotation of triangles)
  typedef struct {
    unsigned int m00;
    unsigned int m01;
    unsigned int m10;
    unsigned int m11;
    unsigned int m20;
    unsigned int m02;
  } moments_t;

  moments_t moments[number_of_colors];
  memset(moments, 0, sizeof(moments_t) * number_of_colors);

  // for each pixel in the (down-sampled) image, check its color and possibly
  // use it to calculate the moments of the ball of said color
  int x_ind, y_ind;
  for (y_ind = start_y; y_ind < end_y; y_ind += downsample) {
    for (x_ind = start_x; x_ind < end_x; x_ind += downsample) {
      Color_t *c = pixel(image, x_ind, y_ind);
      uint8_t ball_color = rgb_to_balls_map[index_from_rgb(*c)];
      *pixel(image_thrs, x_ind, y_ind) = (ball_color + 5) * 45;

      // in case this is either any ball or THE ball, calculate the moments
      if (ball_color != NONE &&
          (only_ball_color == NONE || ball_color == only_ball_color)) {
        // printf("Ball %d found at X: %d, Y: %d!\n", ball_color, x_ind, y_ind);
        moments[ball_color].m00 += 1;
        moments[ball_color].m10 += x_ind;
        moments[ball_color].m01 += y_ind;
        if (compute_orientation) {
          moments[ball_color].m11 += x_ind * y_ind;
          moments[ball_color].m20 += x_ind * x_ind;
          moments[ball_color].m02 += y_ind * y_ind;
        }
      }
    }
  }

  // calculate the center of mass from first moments
  // for all colors
  if (only_ball_color == NONE) {
    for (int i = 0; i < number_of_colors; ++i) {
      if (moments[i].m00 != 0) {
        result[3 * i] = ((float)moments[i].m10) / moments[i].m00;
        result[3 * i + 1] = ((float)moments[i].m01) / moments[i].m00;
        result[3 * i + 2] = NAN;
        /*printf("X: %f, Y: %f, rot: %f from %d occurences, total sum %d/%d, "
               "should be %f and %f\n",
               result[3 * i], result[3 * i + 1], result[3 * i + 2],
               moments[i].m00, moments[i].m01, moments[i].m10,
               ((float)moments[i].m10) / moments[i].m00,
               ((float)moments[i].m01) / moments[i].m00);*/
      } else {
        result[3 * i] = NAN;
        result[3 * i + 1] = NAN;
        result[3 * i + 2] = NAN;
      }
    }
  } else { // only for a specific color
    if (moments[only_ball_color].m00 != 0) {
      result[3 * only_ball_color] =
          ((float)moments[only_ball_color].m10) / moments[only_ball_color].m00;
      result[3 * only_ball_color + 1] =
          ((float)moments[only_ball_color].m01) / moments[only_ball_color].m00;
      result[3 * only_ball_color + 2] = NAN;
      /*printf("now color %d at X: %f, Y: %f\n", only_ball_color,
             result[3 * only_ball_color], result[3 * only_ball_color + 1]);*/

      // calculate theta
      if (compute_orientation) {
        double x = result[3 * only_ball_color];
        double y = result[3 * only_ball_color + 1];
        // fit an ellipse
        float a = ((float)moments[only_ball_color].m20) /
                      moments[only_ball_color].m00 -
                  x * x;
        float b = 2 * (((float)moments[only_ball_color].m11) /
                           moments[only_ball_color].m00 -
                       x * y);
        float c = ((float)moments[only_ball_color].m02) /
                      moments[only_ball_color].m00 -
                  y * y;

        double theta = (1.0 / 2.0) * atan2(b, a - c);

        float minus_sin_theta = -sin(theta);
        float cos_theta = cos(theta);

        float sum_x_tmp = 0;
        float sum_y_tmp = 0;
        for (y_ind = start_y; y_ind < end_y; y_ind += downsample) {
          for (x_ind = start_x; x_ind < end_x; x_ind += downsample) {
            Color_t *c = pixel(image, x_ind, y_ind);
            uint8_t ball_color = rgb_to_balls_map[index_from_rgb(*c)];
            if (ball_color == only_ball_color) {
              float tmp_x =
                  (x_ind - x) * minus_sin_theta + (y_ind - y) * cos_theta;
              float tmp_y =
                  (x_ind - x) * cos_theta - (y_ind - y) * minus_sin_theta;
              sum_x_tmp += tmp_x * tmp_x * tmp_x;
              sum_y_tmp += tmp_y * tmp_y * tmp_y;
            }
          }
        }

        if (abs(sum_x_tmp) > abs(sum_y_tmp)) {
          if (sum_x_tmp > 0) {
            theta += M_PI;
          }
        } else {
          if (sum_y_tmp > 0) {
            theta += M_PI;
          }
        }

        theta += orientation_offset;

        theta = fmod(theta + 2 * M_PI, 2 * M_PI);
        result[3 * only_ball_color + 2] = theta;
      }
    } else {
      result[3 * only_ball_color] = NAN;
      result[3 * only_ball_color + 1] = NAN;
      result[3 * only_ball_color + 2] = NAN;
    }
  }

  return 0;
}

// process the image and return ball positions
// current version only computes orientation if downsample > 1
static int process_image(float result[], Color_t image[], uint8_t image_thrs[],
                         int sx, int sy, int min_size, int max_size,
                         int compute_orientation, float orientation_offset,
                         int downsample, uint8_t mask[], int window_size) {
  int start_x = 0;
  int start_y = 0;
  int end_x = sx;
  int end_y = sy;

  int window_half_size = window_size / 2;

  if (downsample >= 1) {
    int min_size_dwn = min_size / (downsample * downsample);
    int max_size_dwn = max_size / (downsample * downsample);

    // find approximate centers in the downsampled image
    find_location(result, image, image_thrs, sx, sy, min_size_dwn, max_size_dwn,
                  /*compute_orientation=*/0,
                  /*orientation_offset=*/0, downsample, start_x, start_y, end_x,
                  end_y, mask, NONE);

    for (int i = 0; i < number_of_colors; ++i) {
      /*printf("X: %f, Y: %f, color: %d, downsample: %d\n", result[3 * i],
             result[3 * i + 1], i, downsample);*/
    }

    // now look around approximate centers without downsampling for each of the
    // colors
    if (downsample > 1) {
      for (int ball_color = 0; ball_color < number_of_colors; ball_color++) {
        if (isnan(result[3 * ball_color])) {
          continue;
        }
        // find where to look
        start_x = max(result[3 * ball_color + 0] - window_half_size, 0);
        end_x = min(result[3 * ball_color + 0] + window_half_size, sx);
        start_y = max(result[3 * ball_color + 1] - window_half_size, 0);
        end_y = min(result[3 * ball_color + 1] + window_half_size, sy);
        /*printf("Setting start X:%d, Y:%d, end X:%d, Y:%d (ball %d)\n",
         * start_x, start_y, end_x, end_y, ball_color);*/
        find_location(result, image, image_thrs, sx, sy, min_size, max_size,
                      compute_orientation, orientation_offset,
                      /*downsample=*/1, start_x, start_y, end_x, end_y, mask,
                      ball_color);
      }
    }
  } else {
    printf("Downsample must be >= 1");
    return -1;
  }
  return 0;
}

/*
  python wrapper for table init, receives a list of tuples [(h_mid, h_tolerance,
  sat_min, val_min)]
  HSV values should be passed as floats between 0 to 1
*/
static PyObject *PY_init_table(PyObject *dummy, PyObject *args) {
  PyObject *ball_params = NULL;
  uint8_t param_len = 0;
  if (!PyArg_ParseTuple(args, "O", &ball_params)) {
    return NULL;
  }

  Py_INCREF(ball_params);

  param_len = PyList_Size(ball_params);

  PyObject *iter = PyObject_GetIter(ball_params);
  if (!iter) {
    printf("Not an iterator!\n");
  }
  if (PyList_Check(iter)) {
    PyErr_SetString(PyExc_TypeError, "Not a list!");
    Py_XDECREF(ball_params);
    return NULL;
  }

  Ball_t ball_params_data[param_len];
  int i = 0;
  while (1) {
    PyObject *next = PyIter_Next(iter);
    if (!next) {
      break;
    }
    if (!PyTuple_Check(next)) {
      PyErr_SetString(PyExc_TypeError, "Not a tuple!");
      goto fail;
    }
    if (PyTuple_Size(next) != 4) {
      PyErr_SetString(
          PyExc_TypeError,
          "Tuple has an incorrect number of elements (should be four)!");
      goto fail;
    }

    // printf("Got: %f, %f, %f\n", PyFloat_AsDouble(PyTuple_GetItem(next,
    // 0)),PyFloat_AsDouble(PyTuple_GetItem(next,
    // 1)),PyFloat_AsDouble(PyTuple_GetItem(next, 2)));
    double h_mid = PyFloat_AsDouble(PyTuple_GetItem(next, 0));
    double h_tolerance = PyFloat_AsDouble(PyTuple_GetItem(next, 1));

    if (h_mid - h_tolerance < 0) {
      h_mid += 1;
    }

    ball_params_data[i].h_min = (h_mid - h_tolerance) * 360;
    ball_params_data[i].h_max = (h_mid + h_tolerance) * 360;
    ball_params_data[i].sat_min = PyFloat_AsDouble(PyTuple_GetItem(next, 2));
    ball_params_data[i].val_min = PyFloat_AsDouble(PyTuple_GetItem(next, 3));
    ++i;
  }

  init_table(ball_params_data, param_len);

  Py_DECREF(ball_params);
  Py_INCREF(Py_None);
  return Py_None;
fail:
  Py_XDECREF(ball_params);
  return NULL;
}

// python wrapper for find location
static PyObject *PY_find_location(PyObject *dummy, PyObject *args,
                                  PyObject *kw) {
  PyArrayObject *image = NULL;
  PyArrayObject *image_thrs = NULL;
  PyArrayObject *mask = NULL;
  unsigned int min_size = 0, max_size = -1;
  int compute_orientation = 0;
  float result[3 * number_of_colors];
  float orientation_offset = 0;

  static char *keywords[] = {"image",    "image_thrs", "compute_orientation",
                             "size_lim", "mask",       "orientation_offset",
                             NULL};

  // parse the arguments
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|p(II)Of", keywords, &image,
                                   &image_thrs, &compute_orientation, &min_size,
                                   &max_size, &mask, &orientation_offset))
    goto fail;

  // increase pointer reference so that Python doesn't delete it
  Py_INCREF(image);
  Py_INCREF(image_thrs);
  Py_XINCREF(mask);

  long *dims = PyArray_DIMS(image);
  if (PyArray_NDIM(image) != 3) {
    PyErr_SetString(PyExc_TypeError, "Image must have nd==3");
    goto fail;
  }

  if (PyArray_NDIM(image_thrs) != 2) {
    PyErr_SetString(PyExc_TypeError, "Image_thrs must have nd==2");
    goto fail;
  }

  if (!map_initiated) {
    PyErr_SetString(PyExc_TypeError, "You must call init_table() first!");
    goto fail;
  }

  Color_t *img_data = (Color_t *)PyArray_DATA(image);
  uint8_t *img_thrs_data = (uint8_t *)PyArray_DATA(image_thrs);

  find_location(result, img_data, img_thrs_data, dims[0], dims[1], min_size,
                max_size, compute_orientation, orientation_offset,
                /*downsample=*/1,
                /*start_x=*/0, /*start_y=*/0, /*end_x=*/dims[0],
                /*end_y=*/dims[1], /*mask=*/NULL, NONE);
  Py_DECREF(image);
  Py_DECREF(image_thrs);
  Py_XDECREF(mask);
  if (isnan(result[0])) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  PyObject *python_val = PyList_New(number_of_colors);
  for (int i = 0; i < number_of_colors; ++i) {
    PyObject *python_int = Py_BuildValue("(fff)", result[3 * i],
                                         result[3 * i + 1], result[3 * i + 2]);
    PyList_SetItem(python_val, i, python_int);
  }
  return python_val;
fail:
  Py_XDECREF(image);
  Py_XDECREF(image);
  Py_XDECREF(mask);
  return NULL;
}

static PyObject *PY_process_image(PyObject *dummy, PyObject *args,
                                  PyObject *kw) {
  PyArrayObject *image = NULL;
  PyArrayObject *image_thrs = NULL;
  PyArrayObject *mask = NULL;
  unsigned int min_size = 0, max_size = -1;
  int compute_orientation = 0;
  float result[3 * number_of_colors];
  // printf("%d colors\n", number_of_colors);
  unsigned int downsample = 1;
  unsigned int window_size = 64;
  float orientation_offset = 0;

  static char *keywords[] = {
      "image",      "image_thrs",  "compute_orientation", "size_lim", "mask",
      "downsample", "window_size", "orientation_offset",  NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|p(II)OIIf", keywords, &image,
                                   &image_thrs, &compute_orientation, &min_size,
                                   &max_size, &mask, &downsample, &window_size,
                                   &orientation_offset))
    goto fail;

  Py_INCREF(image);
  Py_INCREF(image_thrs);
  Py_XINCREF(mask);

  if (!map_initiated) {
    PyErr_SetString(PyExc_TypeError, "You must call init_table() first!");
    goto fail;
  }

  long *dims = PyArray_DIMS(image);
  if (PyArray_NDIM(image) != 3) {
    PyErr_SetString(PyExc_TypeError, "Image must have nd==3");
    goto fail;
  }

  if (PyArray_NDIM(image_thrs) != 2) {
    PyErr_SetString(PyExc_TypeError, "Image must have nd==2");
    goto fail;
  }

  Color_t *img_data = (Color_t *)PyArray_DATA(image);
  uint8_t *img_thrs_data = (uint8_t *)PyArray_DATA(image_thrs);

  process_image(result, img_data, img_thrs_data, dims[0], dims[1], min_size,
                max_size, compute_orientation, orientation_offset, downsample,
                /*mask=*/NULL, window_size);

  Py_DECREF(image);
  Py_DECREF(image_thrs);
  Py_XDECREF(mask);

  PyObject *python_val = PyList_New(number_of_colors);
  for (int i = 0; i < number_of_colors; ++i) {
    PyObject *python_int = Py_BuildValue("(fff)", result[3 * i],
                                         result[3 * i + 1], result[3 * i + 2]);
    if (isnan(result[3 * i])) {
      Py_INCREF(Py_None);
      PyList_SetItem(python_val, i, Py_None);
    } else {
      /*printf("Center[%d]: %f, %f, %f\n", i, result[3 * i],
                                         result[3 * i + 1], result[3 * i +
         2]);*/
      PyList_SetItem(python_val, i, python_int);
    }
  }
  return python_val;
fail:
  Py_XDECREF(image);
  Py_XDECREF(image_thrs);
  Py_XDECREF(mask);
  return NULL;
}

static PyMethodDef FindObject_Methods[] = {
    {"init_table", (PyCFunction)PY_init_table, METH_VARARGS,
     "Initializes the RGB-BallColor table"},
    {"find_location", (PyCFunction)PY_find_location,
     METH_VARARGS | METH_KEYWORDS,
     "Find object in image and return its location"},
    {"process_image", (PyCFunction)PY_process_image,
     METH_VARARGS | METH_KEYWORDS,
     "Process whole image and return its location"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC PyInit_multi_color_detector(void) {
  PyObject *module;
  static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                         "multi_color_detector",
                                         NULL,
                                         -1,
                                         FindObject_Methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

  module = PyModule_Create(&moduledef);
  if (!module)
    return NULL;

  return module;
}

/* code that makes use of arguments */
/* You will probably need at least
   nd = PyArray_NDIM(<..>)    -- number of dimensions
   dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                 showing length in each dim.
   dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

   If an error occurs goto fail.
 */