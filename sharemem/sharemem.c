#include <Python.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdint.h>

#define SIZE_LIMIT 100

static PyObject * sh_open(PyObject * self, PyObject * args){
    key_t key;
    size_t size;
    int opts=NULL;
    if(!PyArg_ParseTuple(args, "ii|i", &key, &size, &opts)){
        PyErr_SetString(PyExc_TypeError,"Required arguments not found");
        return NULL;
    }

    if (opts==0){
        opts = IPC_CREAT;
    }

    if (size > SIZE_LIMIT) {
        PyErr_SetString(PyExc_TypeError,"Size is bigger than maximum.");
        return NULL;
    }

    int shmid;
    if ((shmid = shmget(key, size, IPC_CREAT | 0666)) < 0) {
        PyErr_SetString(PyExc_RuntimeError,"Syscall to shmget returned errorcode, getmemory failed");
        return NULL;
    }

    void *shm;
    // Now we attach the segment to our data space.
    if ((shm = shmat(shmid, NULL, 0)) == (char *) -1) {
        PyErr_SetString(PyExc_RuntimeError,"Syscall to shmat returned errorcode, attaching failed");
        return NULL;
    }

    memset(shm, 'X', size);

    return Py_BuildValue("i", (unsigned int) shm);
}

static PyObject * sh_read(PyObject * self, PyObject * args){
    void *shm;
    size_t len; 
    if(!PyArg_ParseTuple(args, "ii", &shm, &len)){
        PyErr_SetString(PyExc_TypeError,"Required argument address not found");
        return NULL;
    }

    return Py_BuildValue("y#", shm, len);
}

static PyObject * sh_write(PyObject * self, PyObject * args){
    void *shm;
    Py_buffer b;
    if(!PyArg_ParseTuple(args, "iy*", &shm, &b)){
        PyErr_SetString(PyExc_TypeError,"Required argument address not found");
        return NULL;
    }
    
    memcpy(shm, b.buf, b.len);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * sh_close(PyObject * self, PyObject * args){
    return NULL;
}

static PyMethodDef ShareMem_Methods[] = {
    {"open",  sh_open, METH_VARARGS,
     "Open the shared memory and return pointer"},
    {"read",  sh_read, METH_VARARGS,
     "Read data from the shared memory"},
    {"write",  sh_write, METH_VARARGS,
     "Write data to the shared memory."},
    {"close", sh_close, METH_VARARGS,
     "Read the position from the shared memory with the ith offset."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC PyInit_sharemem(void)
{
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "sharemem",
        NULL,
        -1,
        ShareMem_Methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    (void) PyModule_AddIntConstant(module, "IPC_CREAT",   IPC_CREAT);
    return module;
}
