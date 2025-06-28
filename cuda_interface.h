#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cmath>
#include <random> 
#include <unistd.h> 
#include <map>

extern double TIME_GPU;

extern void simulate(
    float *inputBuffer, 
    float *resultsBuffer, 
    std::map<std::string, float> params, 
    std::map<std::string, int> gpuParams);
