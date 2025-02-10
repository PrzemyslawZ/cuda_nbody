
#include "device_timer.h"

#include <cuda.h>
#include <cuda_runtime.h>

void DeviceTimer::createTimer(){;
    resetTimer();
}

void DeviceTimer::resetTimer(){
    elapsedTime = 0;
}

void DeviceTimer::startTimer(){
    gettimeofday(&t_start, 0);
}

void DeviceTimer::stopTimer(){
    gettimeofday(&t_stop, 0);
    elapsedTime = (1000000.0*(t_stop.tv_sec-t_start.tv_sec) + t_stop.tv_usec-t_start.tv_usec)/1000.0;
}
