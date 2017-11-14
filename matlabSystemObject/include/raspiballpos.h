#ifndef RASPIBALLPOS_H
#define RASPIBALLPOS_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <signal.h>
#include "rtwtypes.h"

struct Position {
   uint16_T  x;
   uint16_T  y;
};
typedef struct Position position;

position* meas_pos;
char_T *shm;
pid_t script_pid;

void raspiballpos_init(uint32_T shm_key, boolean_T execScript, char *script_path, uint32_T fps, uint32_T N);
void raspiballpos_terminate();

void shm_init(uint32_T shm_key);
void shm_terminate();
//void read_pos(uint32_T * x, uint32_T * y, uint8_T len);
void read_pos(uint32_T * data, uint8_T len);

# endif