#ifndef __KERNEL_HANDLER__
#define __KERNEL_HANDLER__
#include "kernel_struct.h"


void runDissipXY_NN(float *newBuffQuantity, float *oldBuffQuantity,
    PhysicalParams params, KernelParams kernelParams);

void runDissipXY(float *newBuffQuantity, float *oldBuffQuantity,
        PhysicalParams params, KernelParams kernelParams);

void runDissipXYZ(float *newBuffQuantity, float *oldBuffQuantity,
    PhysicalParams params, KernelParams kernelParams);

void initializeRandomStates(KernelParams kernelParams);


class KernelHandler{

    public:
        KernelHandler(PhysicalParams, GPUDev);
        ~KernelHandler();

        int modelFact;
        float* mem_hostBuffer[2];

        float* readMemory();
        void run();
        void writeMemory(float*);

        void (*simulFucn)(float *, float *, PhysicalParams, KernelParams);

    private:
        PhysicalParams systemParams;
        KernelParams kernelParams;
        GPUDev gDev;

        float* mem_deviceBuffer[2];

        int nBodies;
        unsigned int memorySize;

        unsigned int hostRead;
        unsigned int memRead;
        unsigned int memWrite;

        unsigned int d_blockSize;
        unsigned int d_numBlocks;

        bool isInitialized=false;
        bool useHostMem;

        void _initialize();
        void _summarize();
        void setModel();
    
        void allocateKernelMemory();
        void freeKernelMemory();

        void allocateMappedMemory(float *[2]);
        void allocateMemory(float *[2]);
        void matchMemory(float *[2], float *[2]);
};

#include "kernel_handler_impl.h"

#endif
