#include "./cuda_interface.h"
#include "./include/kernel_handler.h"
#include "./include/device_timer.h"

KernelHandler *kernel = 0;
DeviceTimer *timer = 0;


struct PhysicalParams PARAMS;
struct GPUDev GPU_PARAMS;


double TIME_GPU = 0;


void loadParams(std::map<std::string, float> params, 
                std::map<std::string, int> gpuParams)
{
    PARAMS.dt = params["dt"];
    PARAMS.Gamma = params["gamma"];
    PARAMS.Jx = params["jx"];
    PARAMS.Jy = params["jy"];
    PARAMS.Jz = params["jz"];
    PARAMS.saveStartPoint = (int)params["save_start"];
	PARAMS.steps = (long)params["tsteps"];
    PARAMS.saveStep = (int)params["save_step"];
	PARAMS.numBodies = (int)params["num_bodies"];
    PARAMS.model = (int)params["model"];

	GPU_PARAMS.blockSize = gpuParams["block_size"];
    GPU_PARAMS.useHostMem = gpuParams["use_host_mem"];
}


void initializeSimulator(float *inputBuffer)
{
    kernel = new KernelHandler(PARAMS, GPU_PARAMS);
    kernel->writeMemory(inputBuffer);
}


void initialize(float *inputBuffer,
                std::map<std::string, float> params, 
                std::map<std::string, int> gpuParams)
{
    loadParams(params, gpuParams);
    initializeSimulator(inputBuffer);

    timer = new DeviceTimer();
    timer->createTimer();
}


void runGPUSimulation(float *resultsBuffer)
{
    int ptrShift = kernel->modelFact * PARAMS.numBodies;
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


void simulate(
    float *inputBuffer, 
    float *resultsBuffer, 
    std::map<std::string, float> params, 
    std::map<std::string, int> gpuParams)
{
    initialize(inputBuffer, params, gpuParams);
    if(kernel)
        runGPUSimulation(resultsBuffer);
    closeSimulation();
}; 