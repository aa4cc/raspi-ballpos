#include <Python.h>
#include "numpy/arrayobject.h"
#include "math.h"

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} Color_t;

#define R 0
#define G 1
#define B 2

#define X 1
#define Y 0

#define min(a,b) ((a<b)?a:b)
#define max(a,b) ((a>b)?a:b)

#define step(endim)
#define pixel(im, x, y) (im+(y*sx)+x)

static int find_location(float result[], Color_t image[], uint8_t image_thrs[],
    int sx, int sy, float color_coefs[], uint8_t threshold,
    int min_size, int max_size, int compute_orientation, float orientation_offset,int downsample, 
	int start_x, int start_y, int end_x, int end_y, uint8_t mask[]){

	unsigned int m00=0, m01=0, m10=0, m11=0, m20=0, m02=0;
	int x_ind, y_ind;
	float x, y, theta=NAN;

    for(y_ind=start_y; y_ind<end_y; y_ind += downsample){ 
        for(x_ind=start_x; x_ind<end_x; x_ind += downsample){

            Color_t *c = pixel(image, x_ind, y_ind);
            float f = c->r*color_coefs[R] + c->g*color_coefs[G] + c->b*color_coefs[B];

            *pixel(image_thrs, x_ind, y_ind) =  (f>threshold)?255:0;
            if(*pixel(image_thrs, x_ind, y_ind)){
                m00 += 1;
                m10 += x_ind;
                m01 += y_ind;
                m11 += x_ind*y_ind; 
                m20 += x_ind*x_ind;
                m02 += y_ind*y_ind; 
            }
        }
        //printf("\n");
    }
    //printf("-------------------------------------------------------------------------------------------------------\n");

    x = ((float) m10)/m00;
    y = ((float) m01)/m00;

    if(compute_orientation){
        float a = ((float)m20)/m00 - x*x;
        float b = 2*(((float)m11)/m00 - x*y);
        float c = ((float)m02)/m00 - y*y;

        theta = (1.0/2.0)*atan2(b, a-c);

        float minus_sin_theta = -sin(theta);
        float cos_theta = cos(theta);

        float sum_tmp=0;
	    for(y_ind=start_y; y_ind<end_y; y_ind += downsample){ 
	        for(x_ind=start_x; x_ind<end_x; x_ind += downsample){
                if(*pixel(image_thrs, x_ind, y_ind)){
                    float tmp = (x_ind-x)*minus_sin_theta + (y_ind-y)*cos_theta;
                    sum_tmp += tmp*tmp*tmp;
                }
            }
        }
        if(sum_tmp>0){
            theta += M_PI;
        }


        theta += orientation_offset;

        theta = fmod(theta+2*M_PI, 2*M_PI);
    }

    result[0] = x;
    result[1] = y;
    result[2] = theta;
	return 0;
}

static int process_image(float result[], Color_t image[], uint8_t image_thrs[],
    int sx, int sy, float color_coefs[], uint8_t threshold,
    int min_size, int max_size, int compute_orientation, float orientation_offset, 
    int downsample,	uint8_t mask[], int window_size){

	int start_x = 0, start_y = 0, end_x=sx, end_y=sy;

	int window_half_size = window_size/2;

	if(downsample>1){
		int min_size_dwn = min_size/(downsample*downsample);
		int max_size_dwn = max_size/(downsample*downsample);

		find_location(result, image, image_thrs, sx, sy, color_coefs, threshold,
			min_size_dwn, max_size_dwn, /*compute_orientation=*/0, /*orientation_offset=*/0,
            downsample, start_x, start_y, end_x, end_y, mask);

		if(isnan(result[0])){
			return-1;
		}

		start_x = max(result[0]-window_half_size, 0);
		end_x   = min(result[0]+window_half_size, sx);
		start_y = max(result[1]-window_half_size, 0);
		end_y   = min(result[1]+window_half_size, sy);

	}

	find_location(result, image, image_thrs, sx, sy, color_coefs, threshold,
	min_size, max_size, compute_orientation, orientation_offset, /*downsample=*/1, 
	start_x, start_y, end_x, end_y, mask);


	if(isnan(result[0])) return-1;
	return 0;
}

static PyObject * PY_find_location(PyObject * dummy, PyObject * args, PyObject * kw)
{
    PyArrayObject *image=NULL;
    PyArrayObject *image_thrs=NULL;
    PyArrayObject *mask=NULL;
    unsigned int min_size=0, max_size=-1;
    int compute_orientation=0;
    uint8_t threshold = 0;
    float coefs[3];
    float result[3];
    float orientation_offset = 0;

    static char *keywords[] = {
    	"image",
    	"image_thrs",
    	"color_coefs",
    	"threshold",
    	"compute_orientation",
    	"size_lim",
    	"mask",
        "orientation_offset",
    	NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO(fff)b|p(II)Of", keywords,
         &image, 
         &image_thrs,
         &coefs[0], &coefs[1], &coefs[2],
         &threshold,
         &compute_orientation,
         &min_size, &max_size,
         &mask,
         &orientation_offset
         ))
        goto fail;
    Py_INCREF(image);
    Py_INCREF(image_thrs);
    Py_XINCREF(mask);

    int *dims = PyArray_DIMS(image);
    if(PyArray_NDIM(image) != 3){
        PyErr_SetString(PyExc_TypeError,"Image must have nd==3");
        goto fail;
    }

    if(PyArray_NDIM(image_thrs) != 2){
        PyErr_SetString(PyExc_TypeError,"Image_thrs must have nd==2");
        goto fail;
    }

    Color_t *img_data = (Color_t *)PyArray_DATA(image);
    uint8_t *img_thrs_data = (uint8_t*) PyArray_DATA(image_thrs);
 
	find_location(result, img_data, img_thrs_data, dims[0], dims[1], coefs, threshold,
		min_size, max_size, compute_orientation, orientation_offset,/*downsample=*/1,
		/*start_x=*/0, /*start_y=*/0, /*end_x=*/dims[0], /*end_y=*/dims[1], /*mask=*/NULL);
    Py_DECREF(image);
    Py_DECREF(image_thrs);
    Py_XDECREF(mask);
    if(isnan(result[0])){
	    Py_INCREF(Py_None);
    	return Py_None;
    }
    return Py_BuildValue("fff", result[0], result[1], result[2]);
 fail:
    Py_XDECREF(image);
    Py_XDECREF(image);
    Py_XDECREF(mask);
    return NULL;
}

static PyObject * PY_process_image(PyObject * dummy, PyObject * args, PyObject * kw)
{
    PyArrayObject *image=NULL;
    PyArrayObject *image_thrs=NULL;
    PyArrayObject *mask=NULL;
    unsigned int min_size=0, max_size=-1;
    int compute_orientation=0;
    uint8_t threshold = 0;
    float coefs[3];
    float result[3];
    unsigned int downsample=1;
    unsigned int window_size=0;
    float orientation_offset=0;

    static char *keywords[] = {
    	"image",
    	"image_thrs",
    	"color_coefs",
    	"threshold",
    	"compute_orientation",
    	"size_lim",
    	"mask",
    	"downsample",
    	"window_size",
        "orientation_offset",
    	NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO(fff)b|p(II)OIIf", keywords,
         &image, 
         &image_thrs,
         &coefs[0], &coefs[1], &coefs[2],
         &threshold,
         &compute_orientation,
         &min_size, &max_size,
         &mask,
         &downsample,
         &window_size,
         &orientation_offset
         ))
        goto fail;

    Py_INCREF(image);
    Py_INCREF(image_thrs);
    Py_XINCREF(mask);

    int *dims = PyArray_DIMS(image);
    if(PyArray_NDIM(image) != 3){
        PyErr_SetString(PyExc_TypeError,"Image must have nd==3");
        goto fail;
    }

    if(PyArray_NDIM(image_thrs) != 2){
        PyErr_SetString(PyExc_TypeError,"Image must have nd==2");
        goto fail;
    }

    Color_t *img_data = (Color_t *)PyArray_DATA(image);
    uint8_t *img_thrs_data = (uint8_t*) PyArray_DATA(image_thrs);
 
	process_image(result, img_data, img_thrs_data, dims[0], dims[1], coefs,
		threshold, min_size, max_size, compute_orientation, orientation_offset,
        downsample, /*mask=*/NULL, window_size);

    Py_DECREF(image);
    Py_DECREF(image_thrs);
    Py_XDECREF(mask);
    if(isnan(result[0])){
	    Py_INCREF(Py_None);
    	return Py_None;
    }
    return Py_BuildValue("fff", result[0], result[1], result[2]);
 fail:
    Py_XDECREF(image);
    Py_XDECREF(image);
    Py_XDECREF(mask);
    return NULL;
}

static PyMethodDef FindObject_Methods[] = {
    {"find_location", (PyCFunction) PY_find_location, METH_VARARGS | METH_KEYWORDS,
     "Find object in image and return its location"},
    {"process_image", (PyCFunction) PY_process_image, METH_VARARGS | METH_KEYWORDS,
     "Process whole image and return its location"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC PyInit_find_object(void)
{
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "find_object",
        NULL,
        -1,
        FindObject_Methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

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