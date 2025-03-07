#include "kernel_handler.h"
#include "nbody_simulation.h"

#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <memory.h>
#include <cuda_runtime.h>
// #include "cuda_runtime_api.h"

void runNbody(float *newPosition, float *oldPosition, float *velocity,
                        float dt, float damping, int nBodies, int blockSize);
void copyMemoryToDevice(float *host, float *device, int numBodies);
void copyMemoryToHost(float *host, float *device, int numBodies);
void allocateMemory(float *velocity[2], int numBodies);
void deleteMemory(float *velocity[2]);
float setE2(float e2);

KernelHandler::KernelHandler(int numBodies, unsigned int numBlocks, unsigned int blockSize): NBodySimulation(numBodies){

    memRead = 0;
    memWrite = 1;
    
    d_blockSize = blockSize;
    d_numBlocks = numBlocks;

    mem_devicePosition[0] = 0;
    mem_devicePosition[1] = 0;
    // mem_deviceVelocity[0] = mem_devicePosition[1] = 0;
    mem_deviceVelocity = 0;

    _initialize(numBodies);
    setPhysicalParams(0.1f, 0.995f);
}

KernelHandler::~KernelHandler(){
    _summarize();
    nBodies = 0;
}

void KernelHandler::run(float dt){

    assert(isInitialized);

    runNbody(
        mem_devicePosition[memRead], 
        mem_devicePosition[memWrite],
        mem_deviceVelocity,
        dt, m_damping, nBodies, d_blockSize);

    std::swap(memRead, memWrite);    
}

void KernelHandler::_initialize(int numBodies) {
    assert(!isInitialized);

    nBodies = numBodies;
    // printf("%d\n", nBodies);
    unsigned int memorySize = sizeof(float) * 4 * numBodies;
    mem_hostPosition = new float[memorySize];
    mem_hostVelocity = new float[memorySize];

    memset(mem_hostPosition, 0, memorySize);
    memset(mem_hostVelocity, 0, memorySize);

    cudaMalloc((void**)&mem_devicePosition[0], memorySize);
    cudaMalloc((void**)&mem_devicePosition[1], memorySize);
    cudaMalloc((void**)&mem_deviceVelocity, memorySize);
    
    isInitialized = true;
}

void KernelHandler::_summarize() {
    assert(isInitialized);

    if(mem_hostPosition)
        delete [] mem_hostPosition;
    if(mem_hostVelocity)
        delete [] mem_hostVelocity;
    if(mem_deviceVelocity)
        cudaFree(mem_deviceVelocity);
    if(mem_devicePosition[0])
        cudaFree(mem_devicePosition[0]);
    if(mem_devicePosition[1])
        cudaFree(mem_devicePosition[1]);


    // deleteMemory(mem_deviceVelocity);
    // deleteMemory(mem_devicePosition);
    // devTimer.resetTimer();
};

void KernelHandler::setPhysicalParams(float damping, float e2){
    m_damping = damping;
    // setE2(e2);
}

float *KernelHandler::readMemory(PhysicalQuantity arr)
{
    assert(isInitialized);

    float *hostData = 0;
    float *deviceData = 0;

    switch(arr)
    {
        case POSITION:
            hostData = mem_hostPosition;
            deviceData = mem_devicePosition[memRead];
            break;
        case VELOCITY:
            hostData = mem_hostVelocity;
            deviceData = mem_deviceVelocity;
            break;
    }
    copyMemoryToHost(hostData, deviceData, nBodies);
    return hostData;
}

void KernelHandler::writeMemory(PhysicalQuantity arr, float *data){
    // assert(isInitialized);
    // copyMemoryToDevice(mem_devicePosition[memW], data, nBodies);
    // copyMemoryToDevice(mem_deviceVelocity, data, nBodies);

    assert(isInitialized);
    float *inData = 0;
    switch(arr)
    {
        case POSITION:
            inData = mem_devicePosition[memWrite];
            break;
        case VELOCITY:
            inData = mem_deviceVelocity;
            break;
    }
    copyMemoryToDevice(inData, data, nBodies);
}
