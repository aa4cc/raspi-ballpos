#include "raspiballpos.h"

void *shm;
pid_t script_pid;
int semaphore;

void raspiballpos_init(uint32_T shm_key, boolean_T execScript, char *script_path, uint32_T fps, uint32_T N, uint32_T devices) {
    printf("Execution of python code select: %d", execScript);
    if (0) {
        script_pid = fork();
        if(script_pid == 0)
        {
            char arg1[10];
            char arg2[10];
            char arg3[10];
            int n;
            n = sprintf(arg1, "-f %d", fps);
            n = sprintf(arg2, "-n %d", N);
            n = sprintf(arg3, "-v");
            
            printf("Running the script\n");
            execl(script_path, "posMeas.py", arg1, arg2, arg3, NULL);
        } else if (script_pid == -1) {
            printf("An error occured during starting the measuring script.\n");
        } else {
            // Give some time to the measuring script to startup
            sleep(8);
        }
    } else {
        script_pid = 0;
    }

    shm_init(shm_key, devices);
}

void raspiballpos_terminate() {
    if (script_pid != 0) {
        printf("-----------------Killing the script-----------------------------\n");
        kill(script_pid, SIGINT);
    }

    shm_terminate();
}

void shm_lock(){
    struct sembuf sb = {0, -1, 0}; /* set to allocate resource */
    if (semop(semaphore, &sb, 1) == -1) {
        fprintf(stderr, "Lock semaphore error");
        return;
    }
}

void shm_unlock(){
    struct sembuf sb = {0, 1, 0}; /* set to deallocate resource */
    if (semop(semaphore, &sb, 1) == -1) {
        fprintf(stderr, "Unlock semaphore error");
        return;
    }
}

void shm_init(uint32_T shm_key, int n) {
    int shmid;
    key_t key;
    key = shm_key;
    
    /*
     * Locate the segment.
     */
    if ((shmid = shmget(key, n*sizeof(Position_t), 0666)) < 0) {
        fprintf(stderr, 
                "An error occured during initialization raspi-ballpos system object.\n %d (shmget)\n Key: %d,\n size: %d,\n mod: 0666\n",
                shmid, key, n*sizeof(Position_t));
        exit(1);
    }
    
    /*
     * Now we attach the segment to our data space.
     */
    if ((shm = shmat(shmid, NULL, 0)) == (char *) -1) {
        fprintf(stderr, "An error occured during initialization raspi-ballpos system object. (shmat)\n");
        exit(1);
    }
    
        /* create a semaphore set with 1 semaphore: */
    if ((semaphore = semget(shm_key, 1, 0666)) == -1) {
        fprintf(stderr, "Syscall to semget returned errorcode, getsemaphore failed");
        return;
    }

    /* initialize semaphore #0 to 1: */
    union semun arg;
    arg.val = 1;
    if (semctl(semaphore, 0, SETVAL, arg) == -1) {
        fprintf(stderr, "Syscall to semctl returned errorcode, setup semaphore failed");
        return;
    }
}

void shm_terminate() {
    /*
     * Detach the segment.
     */
    if (shmdt(shm) == -1) {
        fprintf(stderr, "An error occured during deinitialization raspi-ballpos system object.\n");
        return;
    }
}


void read_pos(void *data, int len){
    // Update Output:
//     int k;
//     Position_t *shm_pos = (Position_t *)shm;
//     Position_t *output = 
//     
//     for(k=0; k<len; k++){
//         data = shm_pos[k].x;
//         data[(k*2)+1] = shm_pos[k].y;
//     }
    shm_lock();
    memcpy(data, shm, len*sizeof(Position_t));
    shm_unlock();
}