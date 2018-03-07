#ifndef RASPIBALLPOS_H
#define RASPIBALLPOS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/wait.h>
#include <signal.h>
#include "rtwtypes.h"

#ifdef __cplusplus
extern "C" {
#endif
    
union semun {
    int              val;    /* Value for SETVAL */
    struct semid_ds *buf;    /* Buffer for IPC_STAT, IPC_SET */
    unsigned short  *array;  /* Array for GETALL, SETALL */
    struct seminfo  *__buf;  /* Buffer for IPC_INFO
                               (Linux specific) */
};

typedef struct{
   float x;
   float y;
   float r;
} Position_t;

void raspiballpos_init(uint32_T shm_key, boolean_T execScript, char *script_path, uint32_T fps, uint32_T N, uint32_T devices);
void raspiballpos_terminate();

void shm_init(uint32_T shm_key, int n);
void shm_terminate();
void read_pos(void *data, int len);

#ifdef __cplusplus
}
#endif

# endif