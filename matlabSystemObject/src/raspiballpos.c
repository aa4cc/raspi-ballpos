#include "raspiballpos.h"

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

void shm_init(uint32_T shm_key, uint32_T n) {
    int shmid;
    key_t key;
    key = shm_key;
    
    /*
     * Locate the segment.
     */
    if ((shmid = shmget(key, n*sizeof(*meas_pos), 0666)) < 0) {
        fprintf(stderr, "An error occured during initialization raspi-ballpos system object.\n %d (shmget)\n Key: %d,\n size: %d,\n mod: 0666\n", shmid, key, 2*sizeof(*meas_pos));
        exit(1);
    }
    
    /*
     * Now we attach the segment to our data space.
     */
    if ((shm = shmat(shmid, NULL, 0)) == (char *) -1) {
        fprintf(stderr, "An error occured during initialization raspi-ballpos system object. (shmat)\n");
        exit(1);
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


void read_pos(uint32_T * data, uint8_T len){
    // Update Output:
    int k = 0;
    position *shm_pos = (position*)shm;
    for(k=0; k<len; k++){
        data[k*2] = shm_pos[k].x;
        data[(k*2)+1] = shm_pos[k].y;
    }
}