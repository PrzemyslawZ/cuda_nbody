#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cmath>
#include <random> // do we need it here ? 
#include <unistd.h> 
#include <map>

#include "./include/kernel_handler.h"
#include "./include/nbodyCPU.h"
#include "./include/device_timer.h"

extern double TIME_GPU, TIME_CPU;
extern std::string PLATFORM;

void simulate(float *resultsBuffer, int buffDim  
              const std::map<std::string, float> params, 
              const std::map<std::string, int> gpuParams)
