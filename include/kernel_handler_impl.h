#include "kernel_handler.h"
#include "nbody_simulation.h"

#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <memory.h>
#include <cuda_runtime.h>
// #include "cuda_runtime_api.h"

void runNbody(float *newPosition, float *oldPosition,
                        float dt, PhysicalParams params, int nBodies, int blockSize);
//void copyMemoryToDevice(float *host, float *device, int numBodies);
//void allocateMemory(float *data[2], int numBodies);
void allocateMemory(float *data[2], int numBodies);
void matchMemory(float *dataDevice[2], float *dataHost[2])
void deleteMemory(float *data[2]);

KernelHandler::KernelHandler(
    PhysicalParams params, 
    unsigned int numBlocks, 
    unsigned int blockSize): NBodySimulation(params.numBodies)
{
    memRead = 0;
    memWrite = 1;
    
    d_blockSize = blockSize;
    d_numBlocks = numBlocks;

    mem_deviceBuffer[0] = 0;
    mem_deviceBuffer[1] = 0;
    
    setPhysicalParams(params);
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
        step, systemParams, nBodies, d_blockSize);

    std::swap(memRead, memWrite);    
}

void KernelHandler::_initialize()
{
    assert(!isInitialized);

    nBodies = systemParams.numBodies;
    unsigned int memorySize = sizeof(float) * 2 * nBodies;

    mem_hostBuffer[0] = new float[memorySize];
    mem_hostBuffer[1] = new float[memorySize];

    memset(mem_hostBuffer[0], 0, memorySize);
    memset(mem_hostBuffer[1], 0, memorySize);

    allocateMemory(mem_deviceBuffer, memorySize);
    matchMemory(mem_deviceBuffer, mem_hostBuffer);

    isInitialized = true;
}
    // cudaMalloc((void**)&mem_deviceBuffer[0], memorySize);
    // cudaMalloc((void**)&mem_deviceBuffer[1], memorySize);

void KernelHandler::_summarize() 
{
    assert(isInitialized);
    if(mem_hostBuffer[0])
        delete [] mem_hostBuffer[0];
    if(mem_hostBuffer[1])
        delete [] mem_hostBuffer[1];
    if(mem_deviceBuffer[0])
        cudaFree(mem_deviceBuffer[0]);
    if(mem_deviceBuffer[1])
        cudaFree(mem_deviceBuffer[1]);
};

void KernelHandler::setPhysicalParams(PhysicalParams params)
{
    systemParams = params;
}

float *KernelHandler::readMemory()
{
    assert(isInitialized);
    float *hostData = mem_hostBuffer[memRead];
    //copyMemoryToHost(hostData, mem_deviceBuffer[memRead], nBodies);
    return hostData;
}

void KernelHandler::writeMemory(float *data)
{
    assert(isInitialized);
    memcpy(mem_hostBuffer[memWrite], data, m_numBodies * 4 * sizeof(T));    
//copyMemoryToDevice(mem_deviceBuffer[memWrite], data, nBodies);
}
