#ifndef __KERNEL_HANDLER__
#define __KERNEL_HANDLER__
#include "nbody_simulation.h"


void runNbody(float *newBuffQuantity, float *oldBuffQuantity,
    PhysicalParams params, KernelParams kernelParams);
void initializeRandomStates(KernelParams kernelParams);
void copyMemoryToDevice(float *host, float *device, int numBodies);
void copyMemoryToHost(float *host, float *device, int numBodies);
void allocateMappedMemory(float * data[2], unsigned int memorySize);
void allocateMemory(float *data[2], unsigned int memorySize);
void matchMemory(float *dataDevice[2], float *dataHost[2]);


class KernelHandler : public NBodySimulation{

    public:
        KernelHandler(PhysicalParams, GPUDev);
        virtual ~KernelHandler();

        float* mem_hostBuffer[2];

        virtual void run(int);
        virtual float* readMemory();
        virtual void writeMemory(float*);

    private:
        KernelHandler() {};
        PhysicalParams systemParams;
        KernelParams kernelParams;
        GPUDev gDev;

        float* mem_deviceBuffer[2];

        bool useHostMem;

        unsigned int hostRead;
        unsigned int memRead;
        unsigned int memWrite;

        unsigned int d_blockSize;
        unsigned int d_numBlocks;
        
        virtual void _initialize();
        virtual void _summarize();
    
        void allocateKernelMemory();
        void freeKernelMemory();
};


#include "kernel_handler_impl.h"

#endif
