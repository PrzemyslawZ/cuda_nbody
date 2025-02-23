#include "kernel_handler.h"
#include "nbody_simulation.h"

#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <memory.h>
#include <cuda_runtime.h>


KernelHandler::KernelHandler(PhysicalParams params, GPUDev gpuParams): NBodySimulation(params)
{
    memRead = 0;
    memWrite = 1;
    systemParams = params;
    gDev = gpuParams;
    
    _initialize();
}

KernelHandler::~KernelHandler()
{
    _summarize();
    nBodies = 0;
}

void KernelHandler::run(int step)
{
    assert(isInitialized);

    runNbody(
        mem_deviceBuffer[memRead], 
        mem_deviceBuffer[memWrite],
        systemParams, kernelParams);

    std::swap(memRead, memWrite);    
}

void KernelHandler::_initialize()
{
    assert(!isInitialized);

    nBodies = systemParams.numBodies;
    useHostMem = gDev.useHostMem;

    kernelParams.blockSize = gDev.blockSize;
    kernelParams.numBlocks = (nBodies + gDev.blockSize - 1) / gDev.blockSize;
    kernelParams.sharedMemSize = gDev.blockSize * 2 * sizeof(float);
    kernelParams.numTiles = (nBodies + gDev.blockSize - 1) / gDev.blockSize;
    
    
    allocateKernelMemory();
    initializeRandomStates(kernelParams);
    isInitialized = true;
}

void KernelHandler::_summarize() 
{
    assert(isInitialized);
    freeKernelMemory();
};

void KernelHandler::allocateKernelMemory()
{
    mem_deviceBuffer[0] = 0;
    mem_deviceBuffer[1] = 0;

    unsigned int memorySize = nBodies * 2 * sizeof(float);
    
    cudaMalloc(
        &kernelParams.rngStates, 
        kernelParams.numBlocks * kernelParams.blockSize * sizeof(curandState));

    if(useHostMem){
        std::cout << "[INFO] Host memory enabled" << std::endl;

        allocateMappedMemory(mem_hostBuffer, memorySize);
        matchMemory(mem_deviceBuffer, mem_hostBuffer);
    }
    else{
        std::cout << "[INFO] Host memory disabled" << std::endl;

        mem_hostBuffer[0] = new float[memorySize];
        memset(mem_hostBuffer[0], 0, memorySize);
        allocateMemory(mem_deviceBuffer, memorySize);
    }
}

void KernelHandler::freeKernelMemory()
{
    if(kernelParams.rngStates)
        cudaFree(kernelParams.rngStates);
    // if(kernelParams.randomValues)
    //     cudaFree(randomValues);

    if(useHostMem){
        if(mem_hostBuffer[0])
            cudaFreeHost(mem_hostBuffer[0]);
        if(mem_hostBuffer[1])
            cudaFreeHost(mem_hostBuffer[1]);
    }
    else{
        if(mem_hostBuffer[0])
            delete [] mem_hostBuffer[0];
        if(mem_hostBuffer[1])
            delete [] mem_hostBuffer[1];
        if(mem_deviceBuffer[0])
            cudaFree(mem_deviceBuffer[0]);
        if(mem_deviceBuffer[1])
            cudaFree(mem_deviceBuffer[1]);
    }
}

float *KernelHandler::readMemory()
{
    assert(isInitialized);
    hostRead =  (useHostMem) ? memRead : 0;
    if(!useHostMem)
        copyMemoryToHost(mem_hostBuffer[hostRead], mem_deviceBuffer[memRead], nBodies);
    return mem_hostBuffer[hostRead];
}

void KernelHandler::writeMemory(float *data)
{
    assert(isInitialized);

    if(useHostMem)
        memcpy(mem_hostBuffer[memWrite], data, 2 * nBodies * sizeof(float));    
    else
        copyMemoryToDevice(mem_deviceBuffer[memWrite], data, nBodies);
}

