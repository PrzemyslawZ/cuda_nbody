#ifndef __DEVICE_TIMER__
#define __DEVICE_TIMER__
// #pragma once


#include <sys/time.h>
#include <stdio.h>

class DeviceTimer{

    public: 
        double elapsedTime;

        DeviceTimer() {};
        virtual ~DeviceTimer() {};

        void startTimer();
        void stopTimer();
        void resetTimer();
        void createTimer();

    private:
        struct timeval t_start, t_stop;
        double start_time;
        double current_time;
};

#include "device_timer_impl.h"

#endif  // __DEVICE_TIMER__