#include <Python.h>
#include "structmember.h"
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/sem.h>

#define SIZE_LIMIT 1000

union semun {
    int              val;    /* Value for SETVAL */
    struct semid_ds *buf;    /* Buffer for IPC_STAT, IPC_SET */
    unsigned short  *array;  /* Array for GETALL, SETALL */
    struct seminfo  *__buf;  /* Buffer for IPC_INFO
                               (Linux specific) */
};

typedef enum{
    UNLOCKED,
    LOCKED
} lock_t;

typedef struct {
    PyObject_HEAD
    unsigned int key;
    int shmid;
    size_t size;
    unsigned int mode;
    void *ptr;
    int semaphore;
    lock_t lock;
    int created;
} SharedMemory;

static int lock(SharedMemory* self)
{
    if(self->lock==LOCKED){
        PyErr_SetString(PyExc_RuntimeError,"Already locked");
        return -1;
    }

    struct sembuf sb = {0, -1, 0}; /* set to allocate resource */
    if (semop(self->semaphore, &sb, 1) == -1) {
        PyErr_SetString(PyExc_RuntimeError,"Lock semaphore error, maybe already deleted?");
        return -1;
    }
    self->lock=LOCKED;
    return 0;
}

static int unlock(SharedMemory* self)
{
    if(self->lock==UNLOCKED){
        PyErr_SetString(PyExc_RuntimeError,"Already unlocked");
        return -1;
    }

    struct sembuf sb = {0, 1, 0}; /* set to deallocate resource */
    if (semop(self->semaphore, &sb, 1) == -1) {
        PyErr_SetString(PyExc_RuntimeError,"Unlock semaphore error");
        return -1;
    }
    self->lock=UNLOCKED;
    return 0;
}

static void
SharedMemory_dealloc(SharedMemory* self)
{
    if(self->ptr){
        //Remove shared memory
        shmdt(self->ptr);
        if(self->created){
            shmctl(self->shmid, IPC_RMID, NULL);
        }
    }
    if(self->semaphore != -1){
        union semun arg;
        if(self->created){
            /* remove semaphore */
            semctl(self->semaphore, 0, IPC_RMID, arg);
        }   
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
SharedMemory_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    SharedMemory *self;

    self = (SharedMemory *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->key=0;
        self->size=0;
        self->shmid=-1;
        self->mode=0666;
        self->ptr = NULL;
        self->semaphore=-1;
        self->lock = LOCKED;
        self->created=1;
    }

    return (PyObject *)self;
}

static int
SharedMemory_init(SharedMemory *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"key", "size", "create", "mode", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "II|pI", kwlist,
                                      &self->key, &self->size, &self->created, &self->mode))
        return -1;

    unsigned int opts = self->mode;
    if (self->created){
        opts |= IPC_CREAT;
    }

    if (self->size > SIZE_LIMIT) {
        char buffer[100];
        sprintf(buffer, "Size is bigger than maximum (size>%d).", SIZE_LIMIT);
        PyErr_SetString(PyExc_ValueError,buffer);
        return -1;
    }

    if(self->size){
        if ((self->shmid = shmget(self->key, self->size, opts)) < 0) {
            PyErr_SetString(PyExc_RuntimeError,"Syscall to shmget returned errorcode, getmemory failed");
            return -1;
        }

        // Now we attach the segment to our data space.
        if ((self->ptr = shmat(self->shmid, NULL, 0)) == (char *) -1) {
            PyErr_SetString(PyExc_RuntimeError,"Syscall to shmat returned errorcode, attaching failed");
            return -1;
        }

        if (self->created){
            memset(self->ptr, 0, self->size);
        }
    }

    /* create a semaphore set with 1 semaphore: */
    if ((self->semaphore = semget(self->key, 1, opts)) == -1) {
        PyErr_SetString(PyExc_RuntimeError,"Syscall to semget returned errorcode, getsemaphore failed");
        return -1;
    }

    /* initialize semaphore #0 to 1: */
    union semun arg;
    arg.val = 1;
    if (semctl(self->semaphore, 0, SETVAL, arg) == -1) {
        PyErr_SetString(PyExc_RuntimeError,"Syscall to semctl returned errorcode, setup semaphore failed");
        return -1;
    }
    self->lock = UNLOCKED;
    return 0;
}


static PyMemberDef SharedMemory_members[] = {
    {"key", T_INT, offsetof(SharedMemory, key), READONLY,
     "Shared memory key number"},
    {"mode", T_INT, offsetof(SharedMemory, mode), READONLY,
     "Shared memory access mode"},
    {"shmid", T_INT, offsetof(SharedMemory, shmid), READONLY,
     "Shared memory access mode"},
    {"ptr", T_INT, offsetof(SharedMemory, ptr), READONLY,
     "Shared memory pointer"},
    {"size", T_INT, offsetof(SharedMemory, size), READONLY,
     "Shared memory size"},
    {"semaphore", T_INT, offsetof(SharedMemory, semaphore), READONLY,
     "Semaphore id"},
    {"created", T_BOOL, offsetof(SharedMemory, created), READONLY,
     "Determine if Shared memory is created by this instance"},
    {"locked", T_BOOL, offsetof(SharedMemory, lock), READONLY,
     "Determine if Shared memory is locked by this instance"},
    {NULL}  /* Sentinel */
};

static PyObject *
SharedMemory_lock(SharedMemory* self)
{
    if(lock(self)<0) return NULL;
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
SharedMemory_unlock(SharedMemory* self)
{
    if(unlock(self)<0) return NULL;
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * SharedMemory_read(SharedMemory* self, PyObject *args, PyObject *kwds){
    size_t length = self->size;
    size_t offset = 0;
    int lockme=1;

    static char *kwlist[] = {"length", "offset", "lock", NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|IIp", kwlist,
                                      &length, &offset, &lockme))
        return NULL;

    if(length+offset > self->size){
        PyErr_SetString(PyExc_IndexError,"Accesing data outside Sharemem area");
        return NULL;
    }

    if(lockme)
        if(lock(self)<0)
            return NULL;

    PyObject *buffer = Py_BuildValue("y#", self->ptr+offset, length);

    if(lockme)
        if(unlock(self)<0)
            return NULL;

    return buffer;
}

static PyObject * SharedMemory_write(SharedMemory* self, PyObject *args, PyObject *kwds){
    size_t offset = 0;
    Py_buffer b;
    int lockme=1;
    static char *kwlist[] = {"data", "offset", "lock", NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "y*|Ip", kwlist,
                                      &b, &offset, &lockme))
        return NULL;
    
    if(b.len+offset > self->size){
        PyErr_SetString(PyExc_IndexError,"Writing data outside Sharemem area");
        return NULL;
    }

    if(lockme)
        if(lock(self)<0)
            return NULL;

    memcpy(self->ptr+offset, b.buf, b.len);

    if(lockme)
        if(unlock(self)<0)
            return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* SharedMemory_repr(PyObject *o){
    char buffer[150];
    SharedMemory *self = (SharedMemory*) o;
    sprintf(buffer, "sharemem.SharedMemory(key=%u, size=%u, create=%s, mode=0o%03o)",self->key, self->size, self->created?"True":"False", self->mode);
    return Py_BuildValue("s", buffer);
}

static PyMethodDef SharedMemory_methods[] = {
    {"lock", (PyCFunction)SharedMemory_lock, METH_NOARGS,
     "Lock IPC lock"
    },
    {"unlock", (PyCFunction)SharedMemory_unlock, METH_NOARGS,
     "Unlock IPC lock"
    },
    {"read", (PyCFunction)SharedMemory_read, METH_KEYWORDS | METH_VARARGS,
     "Return bytes form shared memory with length and size from call arguments, defaults (length=1, offset=0)"
    },
    {"write", (PyCFunction)SharedMemory_write, METH_KEYWORDS | METH_VARARGS,
     "Write data into shared memory, defaults (offset=0)"
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject SharedMemoryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "sharemem.SharedMemory",             /* tp_name */
    sizeof(SharedMemory),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)SharedMemory_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    SharedMemory_repr,         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "SharedMemory objects",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    SharedMemory_methods,             /* tp_methods */
    SharedMemory_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)SharedMemory_init,      /* tp_init */
    0,                         /* tp_alloc */
    SharedMemory_new,                 /* tp_new */
};

static PyModuleDef sharememmodule = {
    PyModuleDef_HEAD_INIT,
    "sharemem",
    "Example module that creates an extension type.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_sharemem(void)

{    PyObject* m;

    if (PyType_Ready(&SharedMemoryType) < 0)
        return NULL;

    m = PyModule_Create(&sharememmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&SharedMemoryType);
    PyModule_AddObject(m, "SharedMemory", (PyObject *)&SharedMemoryType);
    return m;
}