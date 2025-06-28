#include "kernel_handler.h"
#include "kernel_struct.h"

#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <memory.h>
#include <cuda_runtime.h>


KernelHandler::KernelHandler(PhysicalParams params, GPUDev gpuParams)
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
};

void KernelHandler::run()
{
    assert(isInitialized);

    simulFucn(
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
    setModel();

    kernelParams.blockSize = gDev.blockSize;
    kernelParams.numBlocks = (nBodies + gDev.blockSize - 1) / gDev.blockSize;
    kernelParams.sharedMemSize = gDev.blockSize * modelFact * sizeof(float);
    kernelParams.numTiles = (nBodies + gDev.blockSize - 1) / gDev.blockSize;
    
    allocateKernelMemory();
    initializeRandomStates(kernelParams);
    isInitialized = true;
}

void KernelHandler::setModel()
{
    switch(systemParams.model)
    {
        case 1:
            simulFucn = runDissipXY;
            modelFact = 2;
            break;
        case 2:
            simulFucn = runDissipXYZ;
            modelFact = 3;
            break;
        case 3:
            simulFucn = runDissipXY_NN;
            modelFact = 2;
            break;
    };
};

void KernelHandler::_summarize() 
{
    assert(isInitialized);
    freeKernelMemory();
};

void KernelHandler::allocateKernelMemory()
{
    mem_deviceBuffer[0] = 0;
    mem_deviceBuffer[1] = 0;

    memorySize = nBodies * modelFact * sizeof(float);
    
    cudaMalloc(
        &kernelParams.rngStates, 
        kernelParams.numBlocks * kernelParams.blockSize * sizeof(curandState));

    if(useHostMem){
        std::cout << "[INFO] Host memory enabled" << std::endl;
        allocateMappedMemory(mem_hostBuffer);
        matchMemory(mem_deviceBuffer, mem_hostBuffer);
    }
    else{
        std::cout << "[INFO] Host memory disabled" << std::endl;
        mem_hostBuffer[0] = new float[memorySize];
        memset(mem_hostBuffer[0], 0, memorySize);
        allocateMemory(mem_deviceBuffer);
    }
}

void KernelHandler::freeKernelMemory()
{
    if(kernelParams.rngStates)
        cudaFree(kernelParams.rngStates);

    if(useHostMem){
        if(mem_hostBuffer[0])
            cudaFreeHost(mem_hostBuffer[0]);
        if(mem_hostBuffer[1])
            cudaFreeHost(mem_hostBuffer[1]);
    }
    else{
        if(mem_hostBuffer[0])
            delete [] mem_hostBuffer[0];
        if(mem_deviceBuffer[0])
            cudaFree(mem_deviceBuffer[0]);
        if(mem_deviceBuffer[1])
            cudaFree(mem_deviceBuffer[1]);
    }
}

float* KernelHandler::readMemory()
{
    assert(isInitialized);
    hostRead =  (useHostMem) ? memRead : 0;
    if(!useHostMem)
        cudaMemcpy(mem_hostBuffer[hostRead], mem_deviceBuffer[memRead], memorySize, cudaMemcpyDeviceToHost);
    return mem_hostBuffer[hostRead];
}

void KernelHandler::writeMemory(float *data)
{
    assert(isInitialized);
    if(useHostMem)
        memcpy(mem_hostBuffer[memWrite], data, memorySize);  
    else
        cudaMemcpy(mem_deviceBuffer[memWrite], data, memorySize, cudaMemcpyHostToDevice);
    
}

void KernelHandler::matchMemory(float *dataDevice[2], float *dataHost[2])
{
   cudaHostGetDevicePointer((void **)&dataDevice[0], (void *)dataHost[0], 0);
   cudaHostGetDevicePointer((void **)&dataDevice[1], (void *)dataHost[1], 0);
};


void KernelHandler::allocateMappedMemory(float * data[2])
{
    cudaHostAlloc((void **)&data[0], memorySize,
        cudaHostAllocMapped | cudaHostAllocPortable);
    cudaHostAlloc((void **)&data[1], memorySize,
        cudaHostAllocMapped | cudaHostAllocPortable);
}


void KernelHandler::allocateMemory(float *data[2])
{
    cudaMalloc((void**)&data[0], memorySize);
    cudaMalloc((void**)&data[1], memorySize);
}

