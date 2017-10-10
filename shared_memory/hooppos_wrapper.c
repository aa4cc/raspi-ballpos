#include <Python.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdint.h>

struct Position {
   uint16_t  x;
   uint16_t  y;
};
typedef struct Position position;

position *shm_pointer;

static PyObject * measpos_write(PyObject * self, PyObject * args)
{
  int x;
  int y;
  position meas_pos;

  // parse arguments
  if (!PyArg_ParseTuple(args, "ii", &x, &y)) {
    return NULL;
  }

  meas_pos.x = x;
  meas_pos.y = y;

  *shm_pointer = meas_pos;  

  return Py_BuildValue("");
}

static PyObject * measpos_read(PyObject * self, PyObject * args)
{
  PyObject * ret;
  position * meas_pos;

  meas_pos = shm_pointer;

  // build the resulting string into a Python object.
  ret = Py_BuildValue("ii", (int)meas_pos->x, (int)meas_pos->y);

  return ret;
}

static PyObject * predpos_write(PyObject * self, PyObject * args)
{
  int x;
  int y;
  position pred_pos;

  // parse arguments
  if (!PyArg_ParseTuple(args, "ii", &x, &y)) {
    return NULL;
  }

  pred_pos.x = x;
  pred_pos.y = y;

  *(shm_pointer+1) = pred_pos;  

  return Py_BuildValue("");
}

static PyObject * predpos_read(PyObject * self, PyObject * args)
{
  PyObject * ret;
  position * pred_pos;

  pred_pos = (shm_pointer + 1);

  // build the resulting string into a Python object.
  ret = Py_BuildValue("ii", (int)pred_pos->x, (int)pred_pos->y);

  return ret;
}

static PyObject * pos_write(PyObject * self, PyObject * args)
{
  int x;
  int y;
  int i;
  position pos;

  // parse arguments
  if (!PyArg_ParseTuple(args, "iii", &x, &y, &i)) {
    return NULL;
  }

  pos.x = x;
  pos.y = y;

  *(shm_pointer + i) = pos;  

  return Py_BuildValue("");
}

static PyObject * pos_read(PyObject * self, PyObject * args)
{
  int i;
  PyObject * ret;
  position * pos;

  // parse arguments
  if (!PyArg_ParseTuple(args, "i", &i)) {
    return NULL;
  }

  pos = (shm_pointer + i);

  // build the resulting string into a Python object.
  ret = Py_BuildValue("ii", (int)pos->x, (int)pos->y);

  return ret;
}

static PyMethodDef HoopPos_Methods[] = {
    {"measpos_write",  measpos_write, METH_VARARGS,
     "Write the measured position to the shared memory."},
    {"measpos_read",  measpos_read, METH_VARARGS,
     "Read the measured position from the shared memory."},
    {"predpos_write",  predpos_write, METH_VARARGS,
     "Write the predicted position to the shared memory."},
    {"predpos_read",  predpos_read, METH_VARARGS,
     "Read the predicted position from the shared memory."},
    {"pos_write", pos_write, METH_VARARGS,
     "Write the position to the shared memory with the ith offset."},
    {"pos_read", pos_read, METH_VARARGS,
     "Read the position from the shared memory with the ith offset."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC PyInit_hooppos(void)
{
    int shmid;
    key_t key;
    char * shm;

    // key to the segment
    key = 3145914;
    
    // create the segment with the permission to read and write for everyone
    if ((shmid = shmget(key, 5*sizeof(position), IPC_CREAT | 0666)) < 0) {
        perror("shmget");
    }

    // Now we attach the segment to our data space.
    if ((shm = shmat(shmid, NULL, 0)) == (char *) -1) {
        perror("shmat");
    }

    shm_pointer = (position*)shm;

    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "hooppos",
        NULL,
        -1,
        HoopPos_Methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    return module;
}
