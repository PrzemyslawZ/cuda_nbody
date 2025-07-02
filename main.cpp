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

#include "./include/kernel_handler.h"
#include "./include/device_timer.h"

KernelHandler *kernel = 0;
DeviceTimer *timer = 0;

struct PhysicalParams PARAMS;
struct GPUDev GPU_PARAMS;

double TIME_GPU = 0;

void randomizeSystem(float *positions, int numBodies)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> rand01(0, 1);  // Generates 0 or 1

    int posIdx = 0, i = 0;
    while (i < numBodies)
    {
        float2 point;
        point.x = rand() / (float) RAND_MAX * 2 * 3.141;
        point.y = rand() / (float) RAND_MAX * 2 * 3.141;

        int randomSign = 1 - 2 * rand01(gen);
        int randomBit = rand01(gen);
        
        positions[posIdx++] = point.x;
        positions[posIdx++] = point.y;

        i++;
    }
}


void initializeSimulator(float *inputBuffer)
{
    kernel = new KernelHandler(PARAMS, GPU_PARAMS);
    kernel->writeMemory(inputBuffer);
}


void initialize(float *inputBuffer)
{
    initializeSimulator(inputBuffer);
    timer = new DeviceTimer();
    timer->createTimer();
}


void runGPUSimulation(float *resultsBuffer)
{
    int ptrShift = 2 * PARAMS.numBodies;
    int iSave = 0;
    timer->resetTimer();
    timer->startTimer();
    for (int tStep = 0; tStep < PARAMS.steps; tStep++)
    {
        kernel->run();
        if(tStep%PARAMS.saveStep==0 && tStep >= PARAMS.saveStartPoint)
        {
            cudaDeviceSynchronize();
            memcpy(resultsBuffer +  ptrShift*iSave, 
                   kernel->readMemory(), 
                   ptrShift * sizeof(float));
            iSave++;
        }
    } 
    cudaDeviceSynchronize();
    timer->stopTimer();
    TIME_GPU = timer->elapsedTime;
}


void closeSimulation()
{
    if(kernel)
        delete kernel;
    if(timer)
        delete timer;
}


int main()
{

	int numBodies = 1024;
	
    PARAMS.dt = 0.001;
    PARAMS.Gamma = 1.0f;
    PARAMS.Jx = 1.0f;
    PARAMS.Jy = 1.0f;
    PARAMS.Jz = 1.0f;
    PARAMS.saveStartPoint = 0;
	PARAMS.steps = 200;
    PARAMS.saveStep = 10;
	PARAMS.numBodies = numBodies;

	GPU_PARAMS.blockSize = 32;
    GPU_PARAMS.useHostMem = true;
	
	float *inputBuffer = new float[numBodies*2*sizeof(float)];
	float *resultsBuffer = new float[numBodies*2*sizeof(float) * ((PARAMS.steps / PARAMS.saveStep) - PARAMS.saveStartPoint)];
	randomizeSystem(inputBuffer, numBodies);
	
    initialize(inputBuffer);
    if(kernel)
        runGPUSimulation(resultsBuffer);
    closeSimulation();

	if(inputBuffer)
		delete [] inputBuffer;
	if(resultsBuffer)
		delete [] resultsBuffer;
	return 0;
}; 


